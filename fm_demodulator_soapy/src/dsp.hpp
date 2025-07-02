#pragma once

#include <mutex>

#include "types.hpp"

class FMDemodulator {
   private:
    ProcessingState state_;

   public:
    Result process(const IQBuffer& input, const Config& config,
                   AudioBuffer& output);
    const ProcessingState& get_state() const { return state_; }
    void reset() { state_.reset(); }

   private:
    static float apply_filter(float input, float alpha, float& prev_output);
};

template <typename T>
class RingBuffer {
   private:
    std::vector<T> buffer_;
    size_t write_idx_ = 0;
    size_t read_idx_ = 0;
    size_t count_ = 0;
    mutable std::mutex mutex_;

   public:
    explicit RingBuffer(size_t capacity);
    size_t write(const std::vector<T>& input);
    size_t read(std::vector<T>& output, size_t max_samples);
    size_t size() const;
    size_t capacity() const;
    float fill_percent() const;
};