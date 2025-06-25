#ifndef AUDIO_H
#define AUDIO_H

#include <portaudio.h>

#include "common.h"

bool audio_init(void);
void audio_cleanup(void);
float apply_audio_filter(struct audio_filter *filter, float input);
float apply_de_emphasis(float input, float *state);
int audio_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags, void *userData);

#endif