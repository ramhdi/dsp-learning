#include "demodulator.h"

#include <math.h>

#include "audio.h"

static int decimation_counter = 0;

void demodulator_init(void) {
    decimation_counter = 0;
    g_demod_state.prev_phase = 0.0f;
    g_demod_state.dc_i = 0.0f;
    g_demod_state.dc_q = 0.0f;
    g_demod_state.de_emphasis_state = 0.0f;
    g_demod_state.audio_write_idx = 0;
    g_demod_state.audio_read_idx = 0;
    g_demod_state.buffer_count = 0;
}

void fm_demodulate(int16_t *i_samples, int16_t *q_samples, int num_samples) {
    const float dc_alpha = 0.001f;

    for (int n = 0; n < num_samples; n++) {
        float i = (float)i_samples[n] / 32768.0f;
        float q = (float)q_samples[n] / 32768.0f;

        g_demod_state.dc_i =
            dc_alpha * i + (1.0f - dc_alpha) * g_demod_state.dc_i;
        g_demod_state.dc_q =
            dc_alpha * q + (1.0f - dc_alpha) * g_demod_state.dc_q;
        i -= g_demod_state.dc_i;
        q -= g_demod_state.dc_q;

        float phase = atan2f(q, i);
        float phase_diff = phase - g_demod_state.prev_phase;

        if (phase_diff > M_PI) {
            phase_diff -= 2.0f * M_PI;
        } else if (phase_diff < -M_PI) {
            phase_diff += 2.0f * M_PI;
        }

        g_demod_state.prev_phase = phase;

        float audio_sample =
            phase_diff * FM_SAMPLE_RATE / (2.0f * M_PI * 75000.0f);

        decimation_counter++;
        if (decimation_counter >= AUDIO_DECIMATION_FACTOR) {
            decimation_counter = 0;

            audio_sample = apply_de_emphasis(audio_sample,
                                             &g_demod_state.de_emphasis_state);

            if (g_demod_state.buffer_count < AUDIO_BUFFER_SIZE - 1) {
                g_demod_state.audio_buffer[g_demod_state.audio_write_idx] =
                    audio_sample;
                g_demod_state.audio_write_idx =
                    (g_demod_state.audio_write_idx + 1) % AUDIO_BUFFER_SIZE;
                g_demod_state.buffer_count++;
            }
        }
    }
}