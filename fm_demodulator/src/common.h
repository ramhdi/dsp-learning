#ifndef SDR_TYPES_H
#define SDR_TYPES_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MHZ(x) ((long long)(x * 1000000.0 + .5))
#define KHZ(x) ((long long)(x * 1000.0 + .5))

// AUDIO CONFIGURATION CONSTANTS
#define AUDIO_SAMPLE_RATE_HZ 48000
#define AUDIO_CHANNELS_STEREO 2
#define AUDIO_FRAMES_PER_BUFFER 256
#define AUDIO_TEMP_BUFFER_SIZE 512
#define AUDIO_MAIN_BUFFER_SIZE 32768
#define AUDIO_RING_BUFFER_SIZE 81920
#define AUDIO_DEFAULT_AMPLITUDE 0.4f

// SDR HARDWARE CONSTANTS
#define SDR_DEFAULT_FREQ_MHZ 101.1
#define SDR_DEFAULT_SAMPLE_RATE_KHZ 2400
#define SDR_DEFAULT_BANDWIDTH_KHZ 200
#define SDR_DEFAULT_DECIMATION 50
#define SDR_DEFAULT_MANUAL_GAIN_DB 72.0f
#define SDR_BUFFER_SIZE_KIS 640
#define SDR_MAX_IQ_SAMPLES 655360

// FM band limits
#define FM_BAND_MIN_MHZ 70
#define FM_BAND_MAX_MHZ 120

// DSP PROCESSING CONSTANTS
#define DSP_INT16_TO_FLOAT_SCALE 32768.0f
#define DSP_DC_FILTER_ALPHA 0.001f
#define DSP_DEEMPHASIS_ALPHA 0.217f
#define DSP_FM_DEVIATION_HZ 75000.0f

// BUFFER AND STATUS CONSTANTS
#define STATUS_UPDATE_INTERVAL_SEC 2
#define BUFFER_HIGH_THRESHOLD_PCT 80.0f
#define BUFFER_LOW_THRESHOLD_PCT 10.0f
#define MIN_SAMPLES_FOR_LOW_WARNING 2500000ULL

// String buffer sizes
#define TEMP_STRING_BUFFER_SIZE 64

// Unit conversion helpers
#define SAMPLES_TO_MEGASAMPLES 1e6
#define HZ_TO_MHZ 1e6

typedef enum {
    SDR_SUCCESS = 0,
    SDR_ERROR_NO_DATA,
    SDR_ERROR_HARDWARE_FAILURE,
    SDR_ERROR_BUFFER_FULL,
    SDR_ERROR_INVALID_CONFIG,
    SDR_ERROR_INSUFFICIENT_BUFFER
} sdr_result_t;

typedef struct {
    int16_t* i_samples;
    int16_t* q_samples;
    size_t capacity;
    size_t count;
} iq_buffer_t;

typedef struct {
    const int16_t* i_samples;
    const int16_t* q_samples;
    size_t count;
} iq_samples_t;

typedef struct {
    float* samples;
    size_t capacity;
    size_t count;
} audio_buffer_t;

typedef struct {
    float* ring_buffer;
    size_t capacity;
    size_t write_idx;
    size_t read_idx;
    size_t count;
} audio_ring_buffer_t;

typedef struct {
    float prev_phase;
    float dc_i, dc_q;
    float de_emphasis_state;
    uint64_t samples_processed;
    float avg_signal_level;
} processing_state_t;

typedef struct {
    long long center_freq_hz;
    long long sample_rate_hz;
    long long bandwidth_hz;
    const char* rf_port;
    int audio_sample_rate;
    int decimation_factor;
    float max_audio_amplitude;
    const char* gain_control_mode;
    float manual_gain_db;
} sdr_config_t;

typedef struct sdr_interface {
    sdr_result_t (*read_samples)(struct sdr_interface* self,
                                 iq_buffer_t* buffer);
    sdr_result_t (*configure)(struct sdr_interface* self,
                              const sdr_config_t* config);
    void (*cleanup)(struct sdr_interface* self);
    void* impl_data;
} sdr_interface_t;

typedef struct audio_interface {
    sdr_result_t (*write_samples)(struct audio_interface* self,
                                  const audio_buffer_t* buffer);
    sdr_result_t (*start)(struct audio_interface* self);
    void (*stop)(struct audio_interface* self);
    void (*cleanup)(struct audio_interface* self);
    void* impl_data;
} audio_interface_t;

static inline sdr_config_t sdr_config_default(void) {
    return (sdr_config_t){.center_freq_hz = MHZ(SDR_DEFAULT_FREQ_MHZ),
                          .sample_rate_hz = KHZ(SDR_DEFAULT_SAMPLE_RATE_KHZ),
                          .bandwidth_hz = KHZ(SDR_DEFAULT_BANDWIDTH_KHZ),
                          .rf_port = "A_BALANCED",
                          .audio_sample_rate = AUDIO_SAMPLE_RATE_HZ,
                          .decimation_factor = SDR_DEFAULT_DECIMATION,
                          .max_audio_amplitude = AUDIO_DEFAULT_AMPLITUDE,
                          .gain_control_mode = "fast_attack",
                          .manual_gain_db = SDR_DEFAULT_MANUAL_GAIN_DB};
}

static inline sdr_config_t sdr_config_with_frequency(const sdr_config_t* base,
                                                     long long freq_hz) {
    sdr_config_t config = *base;
    config.center_freq_hz = freq_hz;
    return config;
}

static inline iq_samples_t iq_samples_from_buffer(const iq_buffer_t* buffer) {
    return (iq_samples_t){.i_samples = buffer->i_samples,
                          .q_samples = buffer->q_samples,
                          .count = buffer->count};
}

static inline audio_buffer_t audio_buffer_create(float* samples,
                                                 size_t capacity) {
    return (audio_buffer_t){
        .samples = samples, .capacity = capacity, .count = 0};
}

#endif