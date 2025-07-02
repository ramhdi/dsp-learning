#pragma once

#include <chrono>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

// Type aliases for clarity
using Complex = std::complex<float>;
using IQBuffer = std::vector<Complex>;
using AudioBuffer = std::vector<float>;

// AUDIO CONFIGURATION CONSTANTS
constexpr int AUDIO_SAMPLE_RATE_HZ = 48000;
constexpr int AUDIO_CHANNELS_STEREO = 2;
constexpr int AUDIO_FRAMES_PER_BUFFER = 256;
constexpr size_t AUDIO_TEMP_BUFFER_SIZE = 512;
constexpr size_t AUDIO_MAIN_BUFFER_SIZE = 32768;
constexpr size_t AUDIO_RING_BUFFER_SIZE = 81920;
constexpr float AUDIO_DEFAULT_AMPLITUDE = 0.4f;

// SDR HARDWARE CONSTANTS
constexpr double SDR_DEFAULT_FREQ_MHZ = 101.1;
constexpr double SDR_DEFAULT_SAMPLE_RATE_KHZ = 2400;
constexpr double SDR_DEFAULT_BANDWIDTH_KHZ = 200;
constexpr int SDR_DEFAULT_DECIMATION = 50;
constexpr float SDR_DEFAULT_MANUAL_GAIN_DB = 72.0f;
constexpr size_t SDR_BUFFER_SIZE_KIS = 512;
constexpr size_t SDR_MAX_IQ_SAMPLES = 655360;

// FM BAND LIMITS
constexpr int FM_BAND_MIN_MHZ = 70;
constexpr int FM_BAND_MAX_MHZ = 120;

// DSP PROCESSING CONSTANTS
constexpr float DSP_INT16_TO_FLOAT_SCALE = 32768.0f;
constexpr float DSP_DC_FILTER_ALPHA = 0.0001f;
constexpr float DSP_DEEMPHASIS_ALPHA = 0.217f;
constexpr float DSP_FM_DEVIATION_HZ = 75000.0f;

// BUFFER AND STATUS CONSTANTS
constexpr int STATUS_UPDATE_INTERVAL_SEC = 2;
constexpr float BUFFER_HIGH_THRESHOLD_PCT = 80.0f;
constexpr float BUFFER_LOW_THRESHOLD_PCT = 10.0f;
constexpr uint64_t MIN_SAMPLES_FOR_LOW_WARNING = 2500000ULL;

// UNIT CONVERSION HELPERS
constexpr double SAMPLES_TO_MEGASAMPLES = 1e6;
constexpr double HZ_TO_MHZ = 1e6;

// Helper functions for frequency conversion
constexpr double MHz(double x) { return x * 1000000.0; }
constexpr double kHz(double x) { return x * 1000.0; }

// Result codes for operations
enum class Result {
    Success = 0,
    NoData,
    HardwareFailure,
    BufferFull,
    InvalidConfig,
    InsufficientBuffer,
    Timeout,
    StreamError
};

// Processing state for FM demodulation
struct ProcessingState {
    float prev_phase = 0.0f;
    float dc_i = 0.0f, dc_q = 0.0f;
    float de_emphasis_state = 0.0f;
    uint64_t samples_processed = 0;
    float avg_signal_level = 0.0f;
    int decimation_counter = 0;

    void reset() {
        prev_phase = 0.0f;
        dc_i = dc_q = 0.0f;
        de_emphasis_state = 0.0f;
        samples_processed = 0;
        avg_signal_level = 0.0f;
        decimation_counter = 0;
    }
};

// Configuration structure
struct Config {
    // Frequency settings
    double center_freq_hz = MHz(SDR_DEFAULT_FREQ_MHZ);
    double sample_rate_hz = kHz(SDR_DEFAULT_SAMPLE_RATE_KHZ);
    double bandwidth_hz = kHz(SDR_DEFAULT_BANDWIDTH_KHZ);

    // Audio settings
    int audio_sample_rate = AUDIO_SAMPLE_RATE_HZ;
    int decimation_factor = SDR_DEFAULT_DECIMATION;
    float max_audio_amplitude = AUDIO_DEFAULT_AMPLITUDE;

    // Hardware settings
    std::string rf_port = "A_BALANCED";
    std::string gain_control_mode = "fast_attack";
    float manual_gain_db = SDR_DEFAULT_MANUAL_GAIN_DB;

    // Device settings
    std::string device_args = "driver=plutosdr";

    // Validation
    bool is_valid() const {
        return center_freq_hz >= MHz(FM_BAND_MIN_MHZ) &&
               center_freq_hz <= MHz(FM_BAND_MAX_MHZ) && sample_rate_hz > 0 &&
               decimation_factor > 0 && max_audio_amplitude > 0.0f &&
               max_audio_amplitude <= 1.0f;
    }

    // Factory methods
    static Config with_frequency(double freq_hz) {
        Config config;
        config.center_freq_hz = freq_hz;
        return config;
    }

    static Config default_config() { return Config{}; }
};

// Statistics for monitoring
struct Stats {
    uint64_t samples_received = 0;
    uint64_t buffer_overflows = 0;
    uint64_t buffer_underflows = 0;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;

    Stats()
        : start_time(std::chrono::steady_clock::now()),
          last_update(start_time) {}

    double elapsed_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }

    double samples_per_second() const {
        auto elapsed = elapsed_seconds();
        return elapsed > 0 ? samples_received / elapsed : 0.0;
    }

    bool should_update_status() const {
        auto now = std::chrono::steady_clock::now();
        auto since_update =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_update)
                .count();
        return since_update >= STATUS_UPDATE_INTERVAL_SEC;
    }

    void mark_status_updated() {
        last_update = std::chrono::steady_clock::now();
    }
};