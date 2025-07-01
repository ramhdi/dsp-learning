#include "processing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int decimation_counter = 0;

void processing_state_init(processing_state_t* state) {
    memset(state, 0, sizeof(*state));
    decimation_counter = 0;
}

float apply_simple_filter(float input, float alpha, float* prev_output) {
    *prev_output = alpha * input + (1.0f - alpha) * (*prev_output);
    return *prev_output;
}

sdr_result_t fm_demodulate_samples(const iq_samples_t* input,
                                   const sdr_config_t* config,
                                   processing_state_t* state,
                                   audio_buffer_t* output) {
    if (!input || !config || !state || !output) {
        return SDR_ERROR_INVALID_CONFIG;
    }

    output->count = 0;

    for (size_t n = 0; n < input->count; n++) {
        float i = (float)input->i_samples[n] / DSP_INT16_TO_FLOAT_SCALE;
        float q = (float)input->q_samples[n] / DSP_INT16_TO_FLOAT_SCALE;

        state->dc_i = DSP_DC_FILTER_ALPHA * i +
                      (1.0f - DSP_DC_FILTER_ALPHA) * state->dc_i;
        state->dc_q = DSP_DC_FILTER_ALPHA * q +
                      (1.0f - DSP_DC_FILTER_ALPHA) * state->dc_q;
        i -= state->dc_i;
        q -= state->dc_q;

        float phase = atan2f(q, i);
        float phase_diff = phase - state->prev_phase;

        if (phase_diff > M_PI) {
            phase_diff -= (2.0f * M_PI);
        } else if (phase_diff < -M_PI) {
            phase_diff += (2.0f * M_PI);
        }

        state->prev_phase = phase;

        // Convert phase difference to audio sample
        // Scale by sample rate and normalize by FM deviation (75 kHz)
        float audio_sample = phase_diff * config->sample_rate_hz /
                             ((2.0f * M_PI) * DSP_FM_DEVIATION_HZ);

        // Decimation and de-emphasis filtering
        decimation_counter++;
        if (decimation_counter >= config->decimation_factor) {
            decimation_counter = 0;

            // Apply de-emphasis filter (75μs time constant)
            audio_sample = apply_simple_filter(
                audio_sample, DSP_DEEMPHASIS_ALPHA, &state->de_emphasis_state);

            if (output->count >= output->capacity) {
                return SDR_ERROR_BUFFER_FULL;
            }

            // Clamp audio to prevent distortion
            float clamped =
                fmaxf(-config->max_audio_amplitude,
                      fminf(config->max_audio_amplitude, audio_sample));
            output->samples[output->count++] = clamped;
        }

        state->samples_processed++;
    }

    return SDR_SUCCESS;
}

audio_ring_buffer_t create_audio_ring_buffer(size_t capacity) {
    audio_ring_buffer_t ring = {0};
    ring.ring_buffer = calloc(capacity, sizeof(float));
    if (ring.ring_buffer) {
        ring.capacity = capacity;
    }
    return ring;
}

void destroy_audio_ring_buffer(audio_ring_buffer_t* ring) {
    if (ring && ring->ring_buffer) {
        free(ring->ring_buffer);
        memset(ring, 0, sizeof(*ring));
    }
}

sdr_result_t audio_ring_write(audio_ring_buffer_t* ring,
                              const audio_buffer_t* input) {
    if (!ring || !input || !ring->ring_buffer) {
        return SDR_ERROR_INVALID_CONFIG;
    }

    for (size_t i = 0; i < input->count; i++) {
        if (ring->count >= ring->capacity) {
            return SDR_ERROR_BUFFER_FULL;
        }

        ring->ring_buffer[ring->write_idx] = input->samples[i];
        ring->write_idx = (ring->write_idx + 1) % ring->capacity;
        ring->count++;
    }

    return SDR_SUCCESS;
}

size_t audio_ring_read(audio_ring_buffer_t* ring, audio_buffer_t* output) {
    if (!ring || !output || !ring->ring_buffer) {
        return 0;
    }

    size_t samples_to_read =
        (ring->count < output->capacity) ? ring->count : output->capacity;
    output->count = 0;

    for (size_t i = 0; i < samples_to_read; i++) {
        output->samples[output->count++] = ring->ring_buffer[ring->read_idx];
        ring->read_idx = (ring->read_idx + 1) % ring->capacity;
        ring->count--;
    }

    return output->count;
}