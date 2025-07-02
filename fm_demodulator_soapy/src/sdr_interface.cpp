#include "sdr_interface.hpp"

#include <SoapySDR/Formats.hpp>
#include <stdexcept>

SDRInterface::SDRInterface(const std::string& device_args) {
    try {
        device_.reset(SoapySDR::Device::make(device_args));
        if (!device_) {
            throw std::runtime_error("Failed to create SoapySDR device");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("SDR initialization failed: " +
                                 std::string(e.what()));
    }
}

SDRInterface::~SDRInterface() { cleanup(); }

Result SDRInterface::configure(const Config& config) {
    try {
        device_->setSampleRate(SOAPY_SDR_RX, 0, config.sample_rate_hz);
        device_->setFrequency(SOAPY_SDR_RX, 0, config.center_freq_hz);
        device_->setBandwidth(SOAPY_SDR_RX, 0, config.bandwidth_hz);

        if (config.gain_control_mode == "manual") {
            device_->setGain(SOAPY_SDR_RX, 0, config.manual_gain_db);
        } else {
            device_->setGainMode(SOAPY_SDR_RX, 0, true);
        }

        rx_stream_ = device_->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0});
        if (!rx_stream_) {
            return Result::HardwareFailure;
        }

        buffers_.resize(1);

        int ret = device_->activateStream(rx_stream_);
        if (ret != 0) {
            return Result::HardwareFailure;
        }

        return Result::Success;

    } catch (const std::exception&) {
        return Result::HardwareFailure;
    }
}

Result SDRInterface::read_samples(IQBuffer& buffer) {
    if (!rx_stream_) {
        return Result::HardwareFailure;
    }

    buffer.resize(SDR_MAX_IQ_SAMPLES);
    buffers_[0] = buffer.data();

    int flags = 0;
    long long timeNs = 0;

    int ret = device_->readStream(rx_stream_, buffers_.data(), buffer.size(),
                                  flags, timeNs, 100000);

    if (ret == SOAPY_SDR_TIMEOUT) {
        return Result::Timeout;
    }
    if (ret == SOAPY_SDR_OVERFLOW) {
        return Result::BufferFull;
    }
    if (ret < 0) {
        return Result::StreamError;
    }

    buffer.resize(ret);
    return Result::Success;
}

void SDRInterface::cleanup() {
    if (device_ && rx_stream_) {
        try {
            device_->deactivateStream(rx_stream_);
            device_->closeStream(rx_stream_);
        } catch (...) {
        }
        rx_stream_ = nullptr;
    }
}