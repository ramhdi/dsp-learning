#include <atomic>
#include <csignal>
#include <iostream>
#include <string>

#include "audio_interface.hpp"
#include "dsp.hpp"
#include "sdr_interface.hpp"
#include "types.hpp"

static std::atomic<bool> stop_flag{false};

static void handle_signal(int sig) {
    std::cout << "Waiting for process to finish... Got signal " << sig
              << std::endl;
    stop_flag = true;
}

static double parse_frequency(const std::string& freq_str) {
    try {
        double freq_mhz = std::stod(freq_str);
        double freq_hz = MHz(freq_mhz);

        if (freq_hz < MHz(FM_BAND_MIN_MHZ) || freq_hz > MHz(FM_BAND_MAX_MHZ)) {
            std::cerr << "Frequency " << freq_mhz
                      << " MHz is outside typical FM band (" << FM_BAND_MIN_MHZ
                      << "-" << FM_BAND_MAX_MHZ << " MHz)" << std::endl;
            std::cerr << "Continuing anyway..." << std::endl;
        }

        return freq_hz;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid frequency: " + freq_str);
    }
}

static void show_usage(const char* progname) {
    std::cout << "Usage: " << progname
              << " [options] [frequency_MHz] [device_args]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  frequency_MHz  FM station frequency in MHz (default: "
              << SDR_DEFAULT_FREQ_MHZ << ")" << std::endl;
    std::cout
        << "  device_args    SoapySDR device args (default: driver=plutosdr)"
        << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << progname
              << "                              # Use defaults" << std::endl;
    std::cout << "  " << progname
              << " 95.5                         # Tune to 95.5 MHz"
              << std::endl;
    std::cout << "  " << progname
              << " 101.1 \"driver=plutosdr,hostname=192.168.2.1\"" << std::endl;
    std::cout << std::endl;
}

static void print_config(const Config& config) {
    std::cout << "* FM Demodulator for SoapySDR" << std::endl;
    std::cout << "* Frequency: " << config.center_freq_hz / HZ_TO_MHZ << " MHz"
              << std::endl;
    std::cout << "* Sample Rate: "
              << config.sample_rate_hz / SAMPLES_TO_MEGASAMPLES << " MSPS"
              << std::endl;
    std::cout << "* Audio Rate: " << config.audio_sample_rate << " Hz"
              << std::endl;
    std::cout << "* Decimation Factor: " << config.decimation_factor
              << std::endl;
    std::cout << "* Gain Control Mode: " << config.gain_control_mode
              << std::endl;
}

static void print_status(const Stats& stats, const RingBuffer<float>& ring,
                         const ProcessingState& proc_state) {
    float buffer_fill = ring.fill_percent();

    std::cout << "\tRX " << stats.samples_received / SAMPLES_TO_MEGASAMPLES
              << " MSmp, Audio Buffer: " << ring.size() << "/"
              << ring.capacity() << " samples (" << buffer_fill << "%)"
              << std::endl;

    if (buffer_fill > BUFFER_HIGH_THRESHOLD_PCT) {
        std::cout << "\tWarning: Audio buffer nearly full - may cause dropouts"
                  << std::endl;
    } else if (buffer_fill < BUFFER_LOW_THRESHOLD_PCT &&
               stats.samples_received > MIN_SAMPLES_FOR_LOW_WARNING) {
        std::cout << "\tWarning: Audio buffer low - may cause silence"
                  << std::endl;
    }

    std::cout << "\tProcessed: " << proc_state.samples_processed
              << " samples, DC offset: I=" << proc_state.dc_i
              << ", Q=" << proc_state.dc_q << std::endl;
}

int main(int argc, char** argv) {
    try {
        Config config = Config::default_config();

        if (argc > 1) {
            if (std::string(argv[1]) == "-h" ||
                std::string(argv[1]) == "--help") {
                show_usage(argv[0]);
                return 0;
            }

            config.center_freq_hz = parse_frequency(argv[1]);

            if (argc > 2) {
                config.device_args = argv[2];
            }
        }

        if (!config.is_valid()) {
            std::cerr << "Invalid configuration" << std::endl;
            return 1;
        }

        std::signal(SIGINT, handle_signal);

        print_config(config);

        auto sdr = std::make_unique<SDRInterface>(config.device_args);
        auto demod = std::make_unique<FMDemodulator>();
        auto ring = std::make_unique<RingBuffer<float>>(AUDIO_RING_BUFFER_SIZE);
        auto audio = std::make_unique<AudioInterface>(ring.get());

        if (sdr->configure(config) != Result::Success) {
            std::cerr << "Failed to configure SDR" << std::endl;
            return 1;
        }

        if (audio->start() != Result::Success) {
            std::cerr << "Failed to start audio" << std::endl;
            return 1;
        }

        std::cout << "* Starting FM demodulation (press CTRL+C to stop)"
                  << std::endl;
        std::cout << "* Tune a real FM radio to "
                  << config.center_freq_hz / HZ_TO_MHZ
                  << " MHz to verify reception" << std::endl;

        IQBuffer iq_buffer;
        AudioBuffer audio_buffer;
        Stats stats;

        while (!stop_flag) {
            Result result = sdr->read_samples(iq_buffer);
            if (result != Result::Success) {
                if (result == Result::Timeout) continue;
                std::cerr << "Error reading samples" << std::endl;
                break;
            }

            result = demod->process(iq_buffer, config, audio_buffer);
            if (result != Result::Success) {
                if (result == Result::BufferFull) continue;
                std::cerr << "Error in FM demodulation" << std::endl;
                break;
            }

            if (!audio_buffer.empty()) {
                ring->write(audio_buffer);
            }

            stats.samples_received += iq_buffer.size();

            if (stats.should_update_status()) {
                print_status(stats, *ring, demod->get_state());
                stats.mark_status_updated();
            }
        }

        std::cout << "* Shutting down" << std::endl;
        audio->stop();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}