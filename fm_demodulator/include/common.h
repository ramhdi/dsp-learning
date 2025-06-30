#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stdint.h>

#define MHZ(x) ((long long)(x * 1000000.0 + .5))
#define KHZ(x) ((long long)(x * 1000.0 + .5))

#define IIO_ENSURE(expr)                                                  \
    {                                                                     \
        if (!(expr)) {                                                    \
            (void)fprintf(stderr, "assertion failed (%s:%d)\n", __FILE__, \
                          __LINE__);                                      \
            (void)abort();                                                \
        }                                                                 \
    }

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_FRAMES_PER_BUFFER 256
#define IIO_BUFFER_SIZE 1024
#define AUDIO_BUFFER_SIZE 16384
#define MAX_AUDIO_AMPLITUDE 0.4f

#define FM_SAMPLE_RATE MHZ(1.0)
#define DEFAULT_FM_FREQ MHZ(101.1)
#define FM_BANDWIDTH MHZ(0.2)
#define AUDIO_DECIMATION_FACTOR (FM_SAMPLE_RATE / AUDIO_SAMPLE_RATE)

#define RX_GAIN_MODE "fast_attack"
#define RX_MANUAL_GAIN 72.0f

enum iodev { RX, TX };

struct stream_cfg {
    long long bw_hz;
    long long fs_hz;
    long long lo_hz;
    const char *rfport;
};

struct demod_state {
    float prev_phase;
    float *audio_buffer;
    int audio_write_idx;
    int audio_read_idx;
    int buffer_count;
    float dc_i, dc_q;
    float de_emphasis_state;
};

struct audio_filter {
    float alpha;
    float prev_output;
};

extern struct demod_state g_demod_state;
extern bool g_stop;
extern long long g_fm_center_freq;

#endif