#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "io.h"
#include "processing.h"

static volatile bool stop_flag = false;

static void handle_sig(int sig) {
    printf("Waiting for process to finish... Got signal %d\n", sig);
    stop_flag = true;
}

static long long parse_frequency(const char* freq_str) {
    char* endptr;
    double freq_mhz = strtod(freq_str, &endptr);

    if (endptr == freq_str) {
        fprintf(stderr, "Invalid frequency: %s\n", freq_str);
        return -1;
    }

    long long freq_hz = (long long)(freq_mhz * 1000000.0);

    if (freq_hz < MHZ(70) || freq_hz > MHZ(120)) {
        fprintf(stderr,
                "Frequency %.1f MHz is outside typical FM band (70-120 MHz)\n",
                freq_mhz);
        fprintf(stderr, "Continuing anyway...\n");
    }

    return freq_hz;
}

static void show_usage(const char* progname) {
    printf("Usage: %s [options] [frequency_MHz] [pluto_uri]\n", progname);
    printf("\nOptions:\n");
    printf("  frequency_MHz  FM station frequency in MHz (default: 101.1)\n");
    printf("  pluto_uri      PlutoSDR URI (default: auto-detect)\n");
    printf("\nExamples:\n");
    printf("  %s                    # Use default frequency 101.1 MHz\n",
           progname);
    printf("  %s 101.1              # Tune to 101.1 MHz\n", progname);
    printf("  %s 95.5 ip:192.168.2.1  # Tune to 95.5 MHz on specific Pluto\n",
           progname);
    printf("\n");
}

int main(int argc, char** argv) {
    char* pluto_uri = NULL;
    long long center_freq = MHZ(101.1);
    uint64_t samples_received = 0;
    time_t last_status = time(NULL);

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            show_usage(argv[0]);
            return 0;
        }

        long long freq = parse_frequency(argv[1]);
        if (freq > 0) {
            center_freq = freq;
        } else {
            show_usage(argv[0]);
            return 1;
        }

        if (argc > 2) {
            pluto_uri = argv[2];
        }
    }

    signal(SIGINT, handle_sig);

    sdr_config_t config;
    if (center_freq != MHZ(101.1)) {
        sdr_config_t base = sdr_config_default();
        config = sdr_config_with_frequency(&base, center_freq);
    } else {
        config = sdr_config_default();
    }

    printf("* FM Demodulator for ADALM-PLUTO\n");
    printf("* Frequency: %.1f MHz\n", config.center_freq_hz / 1e6);
    printf("* Sample Rate: %.1f MSPS\n", config.sample_rate_hz / 1e6);
    printf("* Audio Rate: %d Hz\n", config.audio_sample_rate);
    printf("* Decimation Factor: %d\n", config.decimation_factor);
    printf("* Gain Control Mode: %s\n", config.gain_control_mode);

    sdr_interface_t* sdr = create_pluto_sdr(pluto_uri);
    if (!sdr) {
        printf("Failed to initialize PLUTO\n");
        return 1;
    }

    sdr_result_t result = sdr->configure(sdr, &config);
    if (result != SDR_SUCCESS) {
        printf("Failed to configure PLUTO streaming\n");
        destroy_sdr_interface(sdr);
        return 1;
    }

    audio_ring_buffer_t ring_buffer = create_audio_ring_buffer(81920);
    if (!ring_buffer.ring_buffer) {
        printf("Failed to create audio ring buffer\n");
        destroy_sdr_interface(sdr);
        return 1;
    }

    audio_interface_t* audio = create_portaudio_output(&ring_buffer);
    if (!audio) {
        printf("Failed to initialize audio\n");
        destroy_audio_ring_buffer(&ring_buffer);
        destroy_sdr_interface(sdr);
        return 1;
    }

    result = audio->start(audio);
    if (result != SDR_SUCCESS) {
        printf("Failed to start audio\n");
        destroy_audio_interface(audio);
        destroy_audio_ring_buffer(&ring_buffer);
        destroy_sdr_interface(sdr);
        return 1;
    }

    processing_state_t processing_state;
    processing_state_init(&processing_state);

    const size_t max_iq_samples = 655360;
    int16_t* i_buffer = malloc(max_iq_samples * sizeof(int16_t));
    int16_t* q_buffer = malloc(max_iq_samples * sizeof(int16_t));
    float* audio_temp = malloc(32768 * sizeof(float));

    if (!i_buffer || !q_buffer || !audio_temp) {
        printf("Failed to allocate buffers\n");
        free(i_buffer);
        free(q_buffer);
        free(audio_temp);
        destroy_audio_interface(audio);
        destroy_audio_ring_buffer(&ring_buffer);
        destroy_sdr_interface(sdr);
        return 1;
    }

    iq_buffer_t iq_buf = {.i_samples = i_buffer,
                          .q_samples = q_buffer,
                          .capacity = max_iq_samples,
                          .count = 0};

    audio_buffer_t audio_buf = audio_buffer_create(audio_temp, 32768);

    printf("* Starting FM demodulation (press CTRL+C to stop)\n");
    printf("* Tune a real FM radio to %.1f MHz to verify reception\n",
           config.center_freq_hz / 1e6);

    while (!stop_flag) {
        result = sdr->read_samples(sdr, &iq_buf);
        if (result != SDR_SUCCESS) {
            printf("Error reading samples: %d\n", result);
            break;
        }

        iq_samples_t iq_samples = iq_samples_from_buffer(&iq_buf);
        result = fm_demodulate_samples(&iq_samples, &config, &processing_state,
                                       &audio_buf);
        if (result != SDR_SUCCESS) {
            printf("Error in FM demodulation: %d\n", result);
            if (result == SDR_ERROR_BUFFER_FULL) {
                continue;
            }
            break;
        }

        if (audio_buf.count > 0) {
            result = audio->write_samples(audio, &audio_buf);
            if (result == SDR_ERROR_BUFFER_FULL) {
                // Audio buffer full, continue processing
            } else if (result != SDR_SUCCESS) {
                printf("Error writing audio samples: %d\n", result);
            }
        }

        samples_received += iq_buf.count;

        time_t now = time(NULL);
        if (now - last_status >= 2) {
            float buffer_fill_percent =
                (float)ring_buffer.count / ring_buffer.capacity * 100.0f;
            printf("\tRX %8.2f MSmp, Audio Buffer: %zu/%zu samples (%.1f%%)\n",
                   samples_received / 1e6, ring_buffer.count,
                   ring_buffer.capacity, buffer_fill_percent);

            if (buffer_fill_percent > 80.0f) {
                printf(
                    "\tWarning: Audio buffer nearly full - may cause "
                    "dropouts\n");
            } else if (buffer_fill_percent < 10.0f &&
                       samples_received > 2500000) {
                printf("\tWarning: Audio buffer low - may cause silence\n");
            }

            printf("\tProcessed: %llu samples, DC offset: I=%.3f, Q=%.3f\n",
                   (unsigned long long)processing_state.samples_processed,
                   processing_state.dc_i, processing_state.dc_q);

            last_status = now;
        }
    }

    printf("* Shutting down\n");
    audio->stop(audio);
    destroy_audio_interface(audio);
    destroy_audio_ring_buffer(&ring_buffer);
    destroy_sdr_interface(sdr);

    free(i_buffer);
    free(q_buffer);
    free(audio_temp);

    return 0;
}