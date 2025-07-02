#include "dsp.hpp"

#include <algorithm>
#include <cmath>

constexpr float PI = 3.14159265358979323846f;

Result FMDemodulator::process(const IQBuffer& input, const Config& config,
                              AudioBuffer& output) {
    if (input.empty()) return Result::NoData;

    output.clear();
    output.reserve(input.size() / config.decimation_factor + 1);

    for (const auto& sample : input) {
        float i = sample.real();
        float q = sample.imag();

        state_.dc_i = apply_filter(i, DSP_DC_FILTER_ALPHA, state_.dc_i);
        state_.dc_q = apply_filter(q, DSP_DC_FILTER_ALPHA, state_.dc_q);
        i -= state_.dc_i;
        q -= state_.dc_q;

        float phase = std::atan2(q, i);
        float phase_diff = phase - state_.prev_phase;

        if (phase_diff > PI) {
            phase_diff -= 2.0f * PI;
        } else if (phase_diff < -PI) {
            phase_diff += 2.0f * PI;
        }

        if (std::isnan(phase_diff)) {
            phase_diff = 0.0f;
        }

        state_.prev_phase = phase;

        float audio_sample = phase_diff * config.sample_rate_hz /
                             (2.0f * PI * DSP_FM_DEVIATION_HZ);

        state_.decimation_counter++;
        if (state_.decimation_counter >= config.decimation_factor) {
            state_.decimation_counter = 0;

            audio_sample = apply_filter(audio_sample, DSP_DEEMPHASIS_ALPHA,
                                        state_.de_emphasis_state);

            float clamped =
                std::clamp(audio_sample, -config.max_audio_amplitude,
                           config.max_audio_amplitude);
            output.push_back(clamped);
        }

        state_.samples_processed++;
    }

    return Result::Success;
}

float FMDemodulator::apply_filter(float input, float alpha,
                                  float& prev_output) {
    prev_output = alpha * input + (1.0f - alpha) * prev_output;
    return prev_output;
}

template <typename T>
RingBuffer<T>::RingBuffer(size_t capacity) : buffer_(capacity) {}

template <typename T>
size_t RingBuffer<T>::write(const std::vector<T>& input) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t written = 0;
    for (const auto& sample : input) {
        if (count_ >= buffer_.size()) {
            break;
        }

        buffer_[write_idx_] = sample;
        write_idx_ = (write_idx_ + 1) % buffer_.size();
        count_++;
        written++;
    }

    return written;
}

template <typename T>
size_t RingBuffer<T>::read(std::vector<T>& output, size_t max_samples) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t to_read = std::min({max_samples, count_, output.size()});
    output.clear();
    output.reserve(to_read);

    for (size_t i = 0; i < to_read; ++i) {
        output.push_back(buffer_[read_idx_]);
        read_idx_ = (read_idx_ + 1) % buffer_.size();
        count_--;
    }

    return to_read;
}

template <typename T>
size_t RingBuffer<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

template <typename T>
size_t RingBuffer<T>::capacity() const {
    return buffer_.size();
}

template <typename T>
float RingBuffer<T>::fill_percent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty() ? 0.0f : (100.0f * count_ / buffer_.size());
}

template class RingBuffer<float>;