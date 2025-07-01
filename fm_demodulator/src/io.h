#ifndef IO_H
#define IO_H

#include "common.h"

sdr_interface_t* create_pluto_sdr(const char* uri);
audio_interface_t* create_portaudio_output(audio_ring_buffer_t* ring_buffer);

void destroy_sdr_interface(sdr_interface_t* sdr);
void destroy_audio_interface(audio_interface_t* audio);

#endif