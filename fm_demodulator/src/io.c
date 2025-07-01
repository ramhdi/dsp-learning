#include "io.h"

#include <iio.h>
#include <math.h>
#include <portaudio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "processing.h"

#define IIO_ENSURE(expr)                                                       \
    {                                                                          \
        if (!(expr)) {                                                         \
            fprintf(stderr, "assertion failed (%s:%d)\n", __FILE__, __LINE__); \
            abort();                                                           \
        }                                                                      \
    }

typedef struct {
    struct iio_context *ctx;
    struct iio_channel *rx0_i;
    struct iio_channel *rx0_q;
    struct iio_buffer *rxbuf;
    int buffer_size_kis;
} pluto_impl_t;

typedef struct {
    PaStream *stream;
    audio_ring_buffer_t *ring_buffer;
    int sample_rate;
    int frames_per_buffer;
} portaudio_impl_t;

enum iodev { RX, TX };

static void errchk(int v, const char *what) {
    if (v < 0) {
        fprintf(stderr, "Error %d writing to channel \"%s\"\n", v, what);
        exit(1);
    }
}

static void wr_ch_lli(struct iio_channel *chn, const char *what,
                      long long val) {
    errchk(iio_channel_attr_write_longlong(chn, what, val), what);
}

static void wr_ch_str(struct iio_channel *chn, const char *what,
                      const char *str) {
    errchk(iio_channel_attr_write(chn, what, str), what);
}

static void wr_ch_double(struct iio_channel *chn, const char *what,
                         double val) {
    errchk(iio_channel_attr_write_double(chn, what, val), what);
}

static char *get_ch_name(const char *type, int id) {
    static char tmpstr[TEMP_STRING_BUFFER_SIZE];
    snprintf(tmpstr, sizeof(tmpstr), "%s%d", type, id);
    return tmpstr;
}

static struct iio_device *get_ad9361_phy(struct iio_context *ctx) {
    struct iio_device *dev = iio_context_find_device(ctx, "ad9361-phy");
    IIO_ENSURE(dev && "No ad9361-phy found");
    return dev;
}

static bool get_ad9361_stream_dev(struct iio_context *ctx, enum iodev d,
                                  struct iio_device **dev) {
    switch (d) {
        case RX:
            *dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
            return *dev != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

static bool get_ad9361_stream_ch(enum iodev d, struct iio_device *dev, int chid,
                                 struct iio_channel **chn) {
    *chn = iio_device_find_channel(dev, get_ch_name("voltage", chid), d == TX);
    if (!*chn)
        *chn = iio_device_find_channel(dev, get_ch_name("altvoltage", chid),
                                       d == TX);
    return *chn != NULL;
}

static bool get_phy_chan(struct iio_context *ctx, enum iodev d, int chid,
                         struct iio_channel **chn) {
    switch (d) {
        case RX:
            *chn = iio_device_find_channel(get_ad9361_phy(ctx),
                                           get_ch_name("voltage", chid), false);
            return *chn != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

static bool get_lo_chan(struct iio_context *ctx, enum iodev d,
                        struct iio_channel **chn) {
    switch (d) {
        case RX:
            *chn = iio_device_find_channel(get_ad9361_phy(ctx),
                                           get_ch_name("altvoltage", 0), true);
            return *chn != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

static bool cfg_ad9361_gain_control(struct iio_context *ctx, enum iodev type,
                                    int chid, const char *gain_mode,
                                    float manual_gain) {
    struct iio_channel *chn = NULL;

    if (!get_phy_chan(ctx, type, chid, &chn)) {
        return false;
    }

    wr_ch_str(chn, "gain_control_mode", gain_mode);

    if (strcmp(gain_mode, "manual") == 0) {
        wr_ch_double(chn, "hardwaregain", manual_gain);
    }

    return true;
}

static sdr_result_t pluto_read_samples_impl(sdr_interface_t *self,
                                            iq_buffer_t *buffer) {
    pluto_impl_t *impl = (pluto_impl_t *)self->impl_data;

    ssize_t nbytes_rx = iio_buffer_refill(impl->rxbuf);
    if (nbytes_rx < 0) {
        return SDR_ERROR_HARDWARE_FAILURE;
    }

    char *p_dat = (char *)iio_buffer_first(impl->rxbuf, impl->rx0_i);
    char *p_end = iio_buffer_end(impl->rxbuf);
    ptrdiff_t p_inc = iio_buffer_step(impl->rxbuf);

    size_t sample_count = 0;
    for (char *p = p_dat; p < p_end; p += p_inc) {
        sample_count++;
    }

    if (sample_count > buffer->capacity) {
        return SDR_ERROR_BUFFER_FULL;
    }

    size_t idx = 0;
    for (p_dat = (char *)iio_buffer_first(impl->rxbuf, impl->rx0_i);
         p_dat < p_end; p_dat += p_inc) {
        buffer->i_samples[idx] = ((int16_t *)p_dat)[0];
        buffer->q_samples[idx] = ((int16_t *)p_dat)[1];
        idx++;
    }

    buffer->count = sample_count;
    return SDR_SUCCESS;
}

static sdr_result_t pluto_configure_impl(sdr_interface_t *self,
                                         const sdr_config_t *config) {
    pluto_impl_t *impl = (pluto_impl_t *)self->impl_data;
    struct iio_channel *chn = NULL;

    if (!get_phy_chan(impl->ctx, RX, 0, &chn)) {
        return SDR_ERROR_HARDWARE_FAILURE;
    }

    wr_ch_str(chn, "rf_port_select", config->rf_port);
    wr_ch_lli(chn, "rf_bandwidth", config->bandwidth_hz);
    wr_ch_lli(chn, "sampling_frequency", config->sample_rate_hz);

    if (!get_lo_chan(impl->ctx, RX, &chn)) {
        return SDR_ERROR_HARDWARE_FAILURE;
    }
    wr_ch_lli(chn, "frequency", config->center_freq_hz);

    if (!cfg_ad9361_gain_control(impl->ctx, RX, 0, config->gain_control_mode,
                                 config->manual_gain_db)) {
        return SDR_ERROR_HARDWARE_FAILURE;
    }

    return SDR_SUCCESS;
}

static void pluto_cleanup_impl(sdr_interface_t *self) {
    pluto_impl_t *impl = (pluto_impl_t *)self->impl_data;

    if (impl->rxbuf) {
        iio_buffer_destroy(impl->rxbuf);
    }
    if (impl->rx0_i) {
        iio_channel_disable(impl->rx0_i);
    }
    if (impl->rx0_q) {
        iio_channel_disable(impl->rx0_q);
    }
    if (impl->ctx) {
        iio_context_destroy(impl->ctx);
    }
    free(impl);
    free(self);
}

static int portaudio_callback(const void *inputBuffer, void *outputBuffer,
                              unsigned long framesPerBuffer,
                              const PaStreamCallbackTimeInfo *timeInfo,
                              PaStreamCallbackFlags statusFlags,
                              void *userData) {
    portaudio_impl_t *impl = (portaudio_impl_t *)userData;
    float *out = (float *)outputBuffer;

    (void)inputBuffer;
    (void)timeInfo;
    (void)statusFlags;

    float temp_buffer[AUDIO_TEMP_BUFFER_SIZE];
    audio_buffer_t output_buf = audio_buffer_create(
        temp_buffer, framesPerBuffer < AUDIO_TEMP_BUFFER_SIZE
                         ? framesPerBuffer
                         : AUDIO_TEMP_BUFFER_SIZE);

    size_t samples_read = audio_ring_read(impl->ring_buffer, &output_buf);

    for (unsigned long i = 0; i < framesPerBuffer; i++) {
        float sample = (i < samples_read) ? output_buf.samples[i] : 0.0f;
        *out++ = sample;  // Left channel
        *out++ = sample;  // Right channel (stereo)
    }

    return paContinue;
}

static sdr_result_t portaudio_write_samples_impl(audio_interface_t *self,
                                                 const audio_buffer_t *buffer) {
    portaudio_impl_t *impl = (portaudio_impl_t *)self->impl_data;
    return audio_ring_write(impl->ring_buffer, buffer);
}

static sdr_result_t portaudio_start_impl(audio_interface_t *self) {
    portaudio_impl_t *impl = (portaudio_impl_t *)self->impl_data;
    PaError err = Pa_StartStream(impl->stream);
    return (err == paNoError) ? SDR_SUCCESS : SDR_ERROR_HARDWARE_FAILURE;
}

static void portaudio_stop_impl(audio_interface_t *self) {
    portaudio_impl_t *impl = (portaudio_impl_t *)self->impl_data;
    if (impl->stream) {
        Pa_StopStream(impl->stream);
    }
}

static void portaudio_cleanup_impl(audio_interface_t *self) {
    portaudio_impl_t *impl = (portaudio_impl_t *)self->impl_data;

    if (impl->stream) {
        Pa_CloseStream(impl->stream);
    }
    Pa_Terminate();
    free(impl);
    free(self);
}

sdr_interface_t *create_pluto_sdr(const char *uri) {
    pluto_impl_t *impl = malloc(sizeof(pluto_impl_t));
    if (!impl) return NULL;

    memset(impl, 0, sizeof(*impl));
    impl->buffer_size_kis = SDR_BUFFER_SIZE_KIS;

    if (uri) {
        impl->ctx = iio_create_context_from_uri(uri);
    } else {
        impl->ctx = iio_create_default_context();
    }

    if (!impl->ctx) {
        free(impl);
        return NULL;
    }

    struct iio_device *rx;
    if (!get_ad9361_stream_dev(impl->ctx, RX, &rx)) {
        iio_context_destroy(impl->ctx);
        free(impl);
        return NULL;
    }

    if (!get_ad9361_stream_ch(RX, rx, 0, &impl->rx0_i) ||
        !get_ad9361_stream_ch(RX, rx, 1, &impl->rx0_q)) {
        iio_context_destroy(impl->ctx);
        free(impl);
        return NULL;
    }

    iio_channel_enable(impl->rx0_i);
    iio_channel_enable(impl->rx0_q);

    impl->rxbuf =
        iio_device_create_buffer(rx, impl->buffer_size_kis * 1024, false);
    if (!impl->rxbuf) {
        iio_context_destroy(impl->ctx);
        free(impl);
        return NULL;
    }

    sdr_interface_t *interface = malloc(sizeof(sdr_interface_t));
    if (!interface) {
        iio_buffer_destroy(impl->rxbuf);
        iio_context_destroy(impl->ctx);
        free(impl);
        return NULL;
    }

    interface->read_samples = pluto_read_samples_impl;
    interface->configure = pluto_configure_impl;
    interface->cleanup = pluto_cleanup_impl;
    interface->impl_data = impl;

    return interface;
}

audio_interface_t *create_portaudio_output(audio_ring_buffer_t *ring_buffer) {
    if (!ring_buffer) return NULL;

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        return NULL;
    }

    portaudio_impl_t *impl = malloc(sizeof(portaudio_impl_t));
    if (!impl) {
        Pa_Terminate();
        return NULL;
    }

    impl->ring_buffer = ring_buffer;
    impl->sample_rate = AUDIO_SAMPLE_RATE_HZ;
    impl->frames_per_buffer = AUDIO_FRAMES_PER_BUFFER;

    PaStreamParameters outputParameters = {0};
    outputParameters.device = Pa_GetDefaultOutputDevice();
    if (outputParameters.device == paNoDevice) {
        free(impl);
        Pa_Terminate();
        return NULL;
    }

    outputParameters.channelCount = AUDIO_CHANNELS_STEREO;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency =
        Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;

    err = Pa_OpenStream(&impl->stream, NULL, &outputParameters,
                        impl->sample_rate, impl->frames_per_buffer, paClipOff,
                        portaudio_callback, impl);

    if (err != paNoError) {
        free(impl);
        Pa_Terminate();
        return NULL;
    }

    audio_interface_t *interface = malloc(sizeof(audio_interface_t));
    if (!interface) {
        Pa_CloseStream(impl->stream);
        free(impl);
        Pa_Terminate();
        return NULL;
    }

    interface->write_samples = portaudio_write_samples_impl;
    interface->start = portaudio_start_impl;
    interface->stop = portaudio_stop_impl;
    interface->cleanup = portaudio_cleanup_impl;
    interface->impl_data = impl;

    return interface;
}

void destroy_sdr_interface(sdr_interface_t *sdr) {
    if (sdr && sdr->cleanup) {
        sdr->cleanup(sdr);
    }
}

void destroy_audio_interface(audio_interface_t *audio) {
    if (audio && audio->cleanup) {
        audio->cleanup(audio);
    }
}