#ifndef SDR_TYPES_H
#define SDR_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MHZ(x) ((long long)(x * 1000000.0 + .5))
#define KHZ(x) ((long long)(x * 1000.0 + .5))

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
    return (sdr_config_t){.center_freq_hz = MHZ(101.1),
                          .sample_rate_hz = KHZ(2400),
                          .bandwidth_hz = KHZ(200),
                          .rf_port = "A_BALANCED",
                          .audio_sample_rate = 48000,
                          .decimation_factor = 50,
                          .max_audio_amplitude = 0.4f,
                          .gain_control_mode = "fast_attack",
                          .manual_gain_db = 72.0f};
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