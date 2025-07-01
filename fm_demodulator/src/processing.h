#ifndef PROCESSING_H
#define PROCESSING_H

#include "common.h"

sdr_result_t fm_demodulate_samples(const iq_samples_t* input,
                                   const sdr_config_t* config,
                                   processing_state_t* state,
                                   audio_buffer_t* output);

float apply_simple_filter(float input, float alpha, float* prev_output);

audio_ring_buffer_t create_audio_ring_buffer(size_t capacity);
void destroy_audio_ring_buffer(audio_ring_buffer_t* ring);
sdr_result_t audio_ring_write(audio_ring_buffer_t* ring,
                              const audio_buffer_t* input);
size_t audio_ring_read(audio_ring_buffer_t* ring, audio_buffer_t* output);

void processing_state_init(processing_state_t* state);

#endif