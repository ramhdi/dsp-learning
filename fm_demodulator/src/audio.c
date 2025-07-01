#include "audio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PaStream *audio_stream = NULL;
static struct audio_filter audio_lpf = {0.5f, 0.0f};

float apply_audio_filter(struct audio_filter *filter, float input) {
    filter->prev_output =
        filter->alpha * input + (1.0f - filter->alpha) * filter->prev_output;
    return filter->prev_output;
}

float apply_de_emphasis(float input, float *state) {
    const float alpha = 0.217f;
    *state = alpha * input + (1.0f - alpha) * (*state);
    return *state;
}

int audio_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags, void *userData) {
    struct demod_state *state = (struct demod_state *)userData;
    float *out = (float *)outputBuffer;

    (void)inputBuffer;
    (void)timeInfo;
    (void)statusFlags;

    for (unsigned long i = 0; i < framesPerBuffer; i++) {
        if (state->buffer_count > 0) {
            float sample = state->audio_buffer[state->audio_read_idx];
            state->audio_read_idx =
                (state->audio_read_idx + 1) % AUDIO_BUFFER_SIZE;
            state->buffer_count--;

            sample = apply_audio_filter(&audio_lpf, sample);
            sample =
                fmaxf(-MAX_AUDIO_AMPLITUDE, fminf(MAX_AUDIO_AMPLITUDE, sample));

            *out++ = sample;
            *out++ = sample;
        } else {
            *out++ = 0.0f;
            *out++ = 0.0f;
        }
    }

    return paContinue;
}

bool audio_init(void) {
    PaError err;

    err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return false;
    }

    g_demod_state.audio_buffer =
        (float *)calloc(AUDIO_BUFFER_SIZE, sizeof(float));
    if (!g_demod_state.audio_buffer) {
        printf("Failed to allocate audio buffer\n");
        return false;
    }

    memset(&g_demod_state, 0, sizeof(g_demod_state));
    g_demod_state.audio_buffer =
        (float *)calloc(AUDIO_BUFFER_SIZE, sizeof(float));

    PaStreamParameters outputParameters;
    outputParameters.device = Pa_GetDefaultOutputDevice();
    if (outputParameters.device == paNoDevice) {
        printf("Error: No default output device.\n");
        return false;
    }

    outputParameters.channelCount = 2;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency =
        Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(&audio_stream, NULL, &outputParameters,
                        AUDIO_SAMPLE_RATE, AUDIO_FRAMES_PER_BUFFER, paClipOff,
                        audio_callback, &g_demod_state);

    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return false;
    }

    err = Pa_StartStream(audio_stream);
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return false;
    }

    printf("* Audio initialized - Sample Rate: %d Hz\n", AUDIO_SAMPLE_RATE);
    printf("* Audio buffer size: %d samples\n", AUDIO_BUFFER_SIZE);
    printf("* Audio decimation factor: %d\n", AUDIO_DECIMATION_FACTOR);
    return true;
}

void audio_cleanup(void) {
    if (audio_stream) {
        Pa_CloseStream(audio_stream);
        Pa_Terminate();
    }

    if (g_demod_state.audio_buffer) {
        free(g_demod_state.audio_buffer);
    }
}