#include "pluto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct iio_context *ctx = NULL;
static struct iio_channel *rx0_i = NULL;
static struct iio_channel *rx0_q = NULL;
static struct iio_buffer *rxbuf = NULL;

void errchk(int v, const char *what) {
    if (v < 0) {
        fprintf(stderr, "Error %d writing to channel \"%s\"\n", v, what);
        pluto_cleanup();
        exit(1);
    }
}

void wr_ch_lli(struct iio_channel *chn, const char *what, long long val) {
    errchk(iio_channel_attr_write_longlong(chn, what, val), what);
}

void wr_ch_str(struct iio_channel *chn, const char *what, const char *str) {
    errchk(iio_channel_attr_write(chn, what, str), what);
}

void wr_ch_double(struct iio_channel *chn, const char *what, double val) {
    errchk(iio_channel_attr_write_double(chn, what, val), what);
}

char *get_ch_name(const char *type, int id) {
    static char tmpstr[64];
    snprintf(tmpstr, sizeof(tmpstr), "%s%d", type, id);
    return tmpstr;
}

struct iio_device *get_ad9361_phy(void) {
    struct iio_device *dev = iio_context_find_device(ctx, "ad9361-phy");
    IIO_ENSURE(dev && "No ad9361-phy found");
    return dev;
}

bool get_ad9361_stream_dev(enum iodev d, struct iio_device **dev) {
    switch (d) {
        case RX:
            *dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
            return *dev != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

bool get_ad9361_stream_ch(enum iodev d, struct iio_device *dev, int chid,
                          struct iio_channel **chn) {
    *chn = iio_device_find_channel(dev, get_ch_name("voltage", chid), d == TX);
    if (!*chn)
        *chn = iio_device_find_channel(dev, get_ch_name("altvoltage", chid),
                                       d == TX);
    return *chn != NULL;
}

bool get_phy_chan(enum iodev d, int chid, struct iio_channel **chn) {
    switch (d) {
        case RX:
            *chn = iio_device_find_channel(get_ad9361_phy(),
                                           get_ch_name("voltage", chid), false);
            return *chn != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

bool get_lo_chan(enum iodev d, struct iio_channel **chn) {
    switch (d) {
        case RX:
            *chn = iio_device_find_channel(get_ad9361_phy(),
                                           get_ch_name("altvoltage", 0), true);
            return *chn != NULL;
        default:
            IIO_ENSURE(0);
            return false;
    }
}

bool cfg_ad9361_gain_control(enum iodev type, int chid) {
    struct iio_channel *chn = NULL;

    printf("* Configuring gain control for channel %d\n", chid);
    if (!get_phy_chan(type, chid, &chn)) {
        printf("Failed to get phy channel for gain control\n");
        return false;
    }

    printf("* Setting gain control mode to: %s\n", RX_GAIN_MODE);
    wr_ch_str(chn, "gain_control_mode", RX_GAIN_MODE);

    if (strcmp(RX_GAIN_MODE, "manual") == 0) {
        printf("* Setting manual gain to: %.1f dB\n", RX_MANUAL_GAIN);
        wr_ch_double(chn, "hardwaregain", RX_MANUAL_GAIN);
    }

    double current_gain;
    int ret = iio_channel_attr_read_double(chn, "hardwaregain", &current_gain);
    if (ret == 0) {
        printf("* Current RX gain: %.1f dB\n", current_gain);
    }

    return true;
}

bool cfg_ad9361_streaming_ch(struct stream_cfg *cfg, enum iodev type,
                             int chid) {
    struct iio_channel *chn = NULL;

    printf("* Acquiring AD9361 phy channel %d\n", chid);
    if (!get_phy_chan(type, chid, &chn)) {
        return false;
    }
    wr_ch_str(chn, "rf_port_select", cfg->rfport);
    wr_ch_lli(chn, "rf_bandwidth", cfg->bw_hz);
    wr_ch_lli(chn, "sampling_frequency", cfg->fs_hz);

    printf("* Acquiring AD9361 RX lo channel\n");
    if (!get_lo_chan(type, &chn)) {
        return false;
    }
    wr_ch_lli(chn, "frequency", cfg->lo_hz);

    if (!cfg_ad9361_gain_control(type, chid)) {
        printf("Failed to configure gain control\n");
        return false;
    }

    return true;
}

bool pluto_init(const char *uri) {
    printf("* Acquiring IIO context\n");
    if (uri) {
        printf("* Using PlutoSDR URI: %s\n", uri);
        ctx = iio_create_context_from_uri(uri);
    } else {
        ctx = iio_create_default_context();
    }

    if (!ctx) {
        printf("No context found\n");
        return false;
    }

    if (iio_context_get_devices_count(ctx) == 0) {
        printf("No devices found\n");
        return false;
    }

    return true;
}

bool pluto_configure_stream(struct stream_cfg *cfg) {
    struct iio_device *rx;

    printf("* Acquiring AD9361 streaming devices\n");
    if (!get_ad9361_stream_dev(RX, &rx)) {
        printf("No rx dev found\n");
        return false;
    }

    printf("* Configuring AD9361 for FM reception\n");
    if (!cfg_ad9361_streaming_ch(cfg, RX, 0)) {
        printf("RX port 0 not found\n");
        return false;
    }

    printf("* Initializing AD9361 IIO streaming channels\n");
    if (!get_ad9361_stream_ch(RX, rx, 0, &rx0_i)) {
        printf("RX chan i not found\n");
        return false;
    }
    if (!get_ad9361_stream_ch(RX, rx, 1, &rx0_q)) {
        printf("RX chan q not found\n");
        return false;
    }

    return true;
}

bool pluto_start_streaming(void) {
    struct iio_device *rx;

    if (!get_ad9361_stream_dev(RX, &rx)) {
        return false;
    }

    printf("* Enabling IIO streaming channels\n");
    iio_channel_enable(rx0_i);
    iio_channel_enable(rx0_q);

    printf("* Creating non-cyclic IIO buffer with %d KiS\n", IIO_BUFFER_SIZE);
    rxbuf = iio_device_create_buffer(rx, IIO_BUFFER_SIZE * 1024, false);
    if (!rxbuf) {
        printf("Could not create RX buffer\n");
        return false;
    }

    return true;
}

ssize_t pluto_read_samples(int16_t **i_samples, int16_t **q_samples) {
    ssize_t nbytes_rx = iio_buffer_refill(rxbuf);
    if (nbytes_rx < 0) {
        printf("Error refilling buf %d\n", (int)nbytes_rx);
        return -1;
    }

    char *p_dat = (char *)iio_buffer_first(rxbuf, rx0_i);
    char *p_end = iio_buffer_end(rxbuf);
    ptrdiff_t p_inc = iio_buffer_step(rxbuf);

    int sample_count = 0;
    for (char *p = p_dat; p < p_end; p += p_inc) {
        sample_count++;
    }

    *i_samples = malloc(sample_count * sizeof(int16_t));
    *q_samples = malloc(sample_count * sizeof(int16_t));

    if (!*i_samples || !*q_samples) {
        if (*i_samples) free(*i_samples);
        if (*q_samples) free(*q_samples);
        return -1;
    }

    int idx = 0;
    for (p_dat = (char *)iio_buffer_first(rxbuf, rx0_i); p_dat < p_end;
         p_dat += p_inc) {
        (*i_samples)[idx] = ((int16_t *)p_dat)[0];
        (*q_samples)[idx] = ((int16_t *)p_dat)[1];
        idx++;
    }

    return sample_count;
}

void pluto_cleanup(void) {
    printf("* Destroying buffers\n");
    if (rxbuf) {
        iio_buffer_destroy(rxbuf);
    }

    printf("* Disabling streaming channels\n");
    if (rx0_i) {
        iio_channel_disable(rx0_i);
    }
    if (rx0_q) {
        iio_channel_disable(rx0_q);
    }

    printf("* Destroying context\n");
    if (ctx) {
        iio_context_destroy(ctx);
    }
}