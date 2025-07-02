#pragma once

#include <SoapySDR/Device.hpp>
#include <memory>

#include "types.hpp"

class SDRInterface {
   private:
    std::unique_ptr<SoapySDR::Device> device_;
    SoapySDR::Stream* rx_stream_ = nullptr;
    std::vector<void*> buffers_;

   public:
    explicit SDRInterface(const std::string& device_args);
    ~SDRInterface();

    Result configure(const Config& config);
    Result read_samples(IQBuffer& buffer);

   private:
    void cleanup();
};