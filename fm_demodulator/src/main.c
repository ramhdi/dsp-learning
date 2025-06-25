#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio.h"
#include "common.h"
#include "demodulator.h"
#include "pluto.h"

struct demod_state g_demod_state;
bool g_stop = false;
long long g_fm_center_freq = DEFAULT_FM_FREQ;

static void handle_sig(int sig) {
    printf("Waiting for process to finish... Got signal %d\n", sig);
    g_stop = true;
}

static void shutdown(void) {
    printf("* Shutting down\n");
    audio_cleanup();
    pluto_cleanup();
    exit(0);
}

static long long parse_frequency(const char *freq_str) {
    char *endptr;
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

static void show_usage(const char *progname) {
    printf("Usage: %s [options] [frequency_MHz] [pluto_uri]\n", progname);
    printf("\nOptions:\n");
    printf("  frequency_MHz  FM station frequency in MHz (default: %.1f)\n",
           DEFAULT_FM_FREQ / 1e6);
    printf("  pluto_uri      PlutoSDR URI (default: auto-detect)\n");
    printf("\nExamples:\n");
    printf("  %s                    # Use default frequency %.1f MHz\n",
           progname, DEFAULT_FM_FREQ / 1e6);
    printf("  %s 101.1              # Tune to 101.1 MHz\n", progname);
    printf("  %s 95.5 ip:192.168.2.1  # Tune to 95.5 MHz on specific Pluto\n",
           progname);
    printf("\n");
}

int main(int argc, char **argv) {
    char *pluto_uri = NULL;
    size_t nrx = 0;

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            show_usage(argv[0]);
            return 0;
        }

        long long freq = parse_frequency(argv[1]);
        if (freq > 0) {
            g_fm_center_freq = freq;
        } else {
            show_usage(argv[0]);
            return 1;
        }

        if (argc > 2) {
            pluto_uri = argv[2];
        }
    }

    signal(SIGINT, handle_sig);

    struct stream_cfg rxcfg = {.bw_hz = FM_BANDWIDTH,
                               .fs_hz = FM_SAMPLE_RATE,
                               .lo_hz = g_fm_center_freq,
                               .rfport = "A_BALANCED"};

    printf("* FM Demodulator for ADALM-PLUTO\n");
    printf("* Frequency: %.1f MHz\n", g_fm_center_freq / 1e6);
    printf("* Sample Rate: %.1f MSPS\n", FM_SAMPLE_RATE / 1e6);
    printf("* Audio Rate: %d Hz\n", AUDIO_SAMPLE_RATE);
    printf("* Gain Control Mode: %s\n", RX_GAIN_MODE);

    if (!pluto_init(pluto_uri)) {
        printf("Failed to initialize PLUTO\n");
        return 1;
    }

    if (!pluto_configure_stream(&rxcfg)) {
        printf("Failed to configure PLUTO streaming\n");
        shutdown();
    }

    if (!pluto_start_streaming()) {
        printf("Failed to start PLUTO streaming\n");
        shutdown();
    }

    if (!audio_init()) {
        printf("Failed to initialize audio\n");
        shutdown();
    }

    demodulator_init();

    printf("* Starting FM demodulation (press CTRL+C to stop)\n");
    printf("* Tune a real FM radio to %.1f MHz to verify reception\n",
           g_fm_center_freq / 1e6);

    while (!g_stop) {
        int16_t *i_samples, *q_samples;
        ssize_t sample_count = pluto_read_samples(&i_samples, &q_samples);

        if (sample_count > 0) {
            fm_demodulate(i_samples, q_samples, sample_count);
            free(i_samples);
            free(q_samples);

            nrx += sample_count;
            if ((nrx / 1000000) % 5 == 0) {
                printf("\tRX %8.2f MSmp, Audio Buffer: %d samples\n", nrx / 1e6,
                       g_demod_state.buffer_count);
            }
        } else {
            printf("Error reading samples\n");
            break;
        }
    }

    shutdown();
    return 0;
}