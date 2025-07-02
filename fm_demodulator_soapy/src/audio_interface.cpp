#include "audio_interface.hpp"

#include <stdexcept>

AudioInterface::AudioInterface(RingBuffer<float>* ring_buffer)
    : ring_buffer_(ring_buffer) {
    if (!ring_buffer_) {
        throw std::invalid_argument("Ring buffer cannot be null");
    }

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error("Failed to initialize PortAudio: " +
                                 std::string(Pa_GetErrorText(err)));
    }
    initialized_ = true;

    PaStreamParameters outputParameters = {};
    outputParameters.device = Pa_GetDefaultOutputDevice();
    if (outputParameters.device == paNoDevice) {
        throw std::runtime_error("No default audio output device");
    }

    outputParameters.channelCount = AUDIO_CHANNELS_STEREO;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency =
        Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;

    err = Pa_OpenStream(&stream_, nullptr, &outputParameters,
                        AUDIO_SAMPLE_RATE_HZ, AUDIO_FRAMES_PER_BUFFER,
                        paClipOff, callback, this);

    if (err != paNoError) {
        throw std::runtime_error("Failed to open audio stream: " +
                                 std::string(Pa_GetErrorText(err)));
    }
}

AudioInterface::~AudioInterface() { cleanup(); }

Result AudioInterface::start() {
    if (!stream_) return Result::HardwareFailure;

    PaError err = Pa_StartStream(stream_);
    return (err == paNoError) ? Result::Success : Result::HardwareFailure;
}

void AudioInterface::stop() {
    if (stream_) {
        Pa_StopStream(stream_);
    }
}

void AudioInterface::cleanup() {
    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
    if (initialized_) {
        Pa_Terminate();
        initialized_ = false;
    }
}

int AudioInterface::callback(const void* inputBuffer, void* outputBuffer,
                             unsigned long framesPerBuffer,
                             const PaStreamCallbackTimeInfo* timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void* userData) {
    (void)inputBuffer;
    (void)timeInfo;
    (void)statusFlags;

    auto* self = static_cast<AudioInterface*>(userData);
    auto* out = static_cast<float*>(outputBuffer);

    std::vector<float> temp_buffer(framesPerBuffer);
    size_t samples_read =
        self->ring_buffer_->read(temp_buffer, framesPerBuffer);

    for (unsigned long i = 0; i < framesPerBuffer; ++i) {
        float sample = (i < samples_read) ? temp_buffer[i] : 0.0f;
        *out++ = sample;
        *out++ = sample;
    }

    return paContinue;
}