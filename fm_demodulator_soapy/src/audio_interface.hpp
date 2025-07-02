#pragma once

#include <portaudio.h>

#include "dsp.hpp"
#include "types.hpp"

class AudioInterface {
   private:
    PaStream* stream_ = nullptr;
    RingBuffer<float>* ring_buffer_;
    bool initialized_ = false;

   public:
    explicit AudioInterface(RingBuffer<float>* ring_buffer);
    ~AudioInterface();

    Result start();
    void stop();

   private:
    void cleanup();
    static int callback(const void* inputBuffer, void* outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo* timeInfo,
                        PaStreamCallbackFlags statusFlags, void* userData);
};