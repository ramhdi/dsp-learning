# ADALM-PLUTO FM Demodulator

Real-time FM radio demodulator using ADALM-PLUTO SDR with audio playback, built with Data-Oriented Programming principles.

## Features

- Real-time FM demodulation using phase discriminator
- Audio playback via PortAudio
- Configurable FM station tuning
- Clean data-oriented architecture
- Explicit error handling and data flow
- Cross-platform support

## Architecture

Built using Data-Oriented Programming (DOP) principles for improved maintainability and testability:

- **Explicit data flow**: Clear IQ → Processing → Audio pipeline
- **No global state**: All state passed explicitly
- **Pure processing functions**: DSP isolated from I/O
- **Interface abstraction**: Hardware abstracted behind clean APIs

## Project Structure

```
fm_demodulator/
├── CMakeLists.txt
├── README.md
├── include/
│   └── sdr_fm.h           # Public API
└── src/
    ├── common.h           # Data structures
    ├── processing.h/.c    # Pure DSP functions
    ├── io.h/.c            # Hardware interfaces
    └── main.c             # Application entry
```

## Build Instructions (MSYS2)

### Dependencies

```bash
pacman -S mingw-w64-ucrt-x86_64-toolchain
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
pacman -S mingw-w64-ucrt-x86_64-libiio
pacman -S mingw-w64-ucrt-x86_64-portaudio
```

### Build

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

## Usage

```bash
# Default frequency (101.1 MHz)
./fm_demodulator

# Specific frequency
./fm_demodulator 95.5

# Specific PlutoSDR
./fm_demodulator 101.1 ip:192.168.2.1
```

## Configuration

### Audio Configuration

| Constant                  | Value | Purpose                                                                         | Configurable Range   |
| ------------------------- | ----- | ------------------------------------------------------------------------------- | -------------------- |
| `AUDIO_SAMPLE_RATE_HZ`    | 48000 | Standard professional audio sample rate, supported by all modern audio hardware | 44100-96000 Hz       |
| `AUDIO_CHANNELS_STEREO`   | 2     | Stereo output (mono signal duplicated to both channels)                         | Fixed                |
| `AUDIO_FRAMES_PER_BUFFER` | 256   | Low-latency buffer size, ~5.3ms at 48kHz. Balances latency vs. CPU overhead     | 64-1024 samples      |
| `AUDIO_TEMP_BUFFER_SIZE`  | 512   | Temporary buffer for PortAudio callback, must be ≥ frames per buffer            | 256-2048 samples     |
| `AUDIO_MAIN_BUFFER_SIZE`  | 32768 | Main processing buffer size, ~680ms at 48kHz                                    | 8192-65536 samples   |
| `AUDIO_RING_BUFFER_SIZE`  | 81920 | Ring buffer provides ~1.7 second audio buffering to handle processing jitter    | 16384-262144 samples |
| `AUDIO_DEFAULT_AMPLITUDE` | 0.4   | Volume control to prevent clipping and provide comfortable listening level      | 0.1-0.8              |

### SDR Hardware Configuration

| Constant                      | Value  | Purpose                                                                   | Configurable Range |
| ----------------------------- | ------ | ------------------------------------------------------------------------- | ------------------ |
| `SDR_DEFAULT_FREQ_MHZ`        | 101.1  | Common FM frequency with good reception in most areas                     | 70-120 MHz         |
| `SDR_DEFAULT_SAMPLE_RATE_KHZ` | 2400   | PlutoSDR minimum stable sample rate, provides 200kHz FM channel bandwidth | 2400-61440 kHz     |
| `SDR_DEFAULT_BANDWIDTH_KHZ`   | 200    | Standard FM broadcast channel bandwidth (±75kHz deviation + guard bands)  | 150-300 kHz        |
| `SDR_DEFAULT_DECIMATION`      | 50     | Decimation factor: 2400kHz → 48kHz (2400/50 = 48)                         | 25-100             |
| `SDR_DEFAULT_MANUAL_GAIN_DB`  | 72.0   | High gain setting for good weak signal sensitivity                        | 0-77 dB            |
| `SDR_BUFFER_SIZE_KIS`         | 640    | PlutoSDR buffer size in kilosamples (640K samples = ~267ms at 2.4MSPS)    | 64-16384 KiS       |
| `SDR_MAX_IQ_SAMPLES`          | 655360 | Maximum IQ samples per processing buffer, matches SDR buffer capacity     | 65536-1048576      |

### FM Band Limits

| Constant          | Value | Purpose                                                                  | Configurable Range |
| ----------------- | ----- | ------------------------------------------------------------------------ | ------------------ |
| `FM_BAND_MIN_MHZ` | 70    | Lower limit of global FM broadcast band (Region 2: 88-108, Japan: 76-95) | 70-88 MHz          |
| `FM_BAND_MAX_MHZ` | 120   | Upper limit accommodating all global FM bands plus some margin           | 108-120 MHz        |

### DSP Processing Constants

| Constant                   | Value   | Purpose                                                                             | Configurable Range |
| -------------------------- | ------- | ----------------------------------------------------------------------------------- | ------------------ |
| `DSP_INT16_TO_FLOAT_SCALE` | 32768.0 | Converts 16-bit signed integers (-32768 to +32767) to normalized floats (±1.0)      | Fixed              |
| `DSP_DC_FILTER_ALPHA`      | 0.001   | DC offset removal filter coefficient. Low value = slow adaptation, stable operation | 0.0001-0.01        |
| `DSP_DEEMPHASIS_ALPHA`     | 0.217   | 75μs de-emphasis time constant for North American FM standard (α = 1-e^(-1/τf))     | Fixed for standard |
| `DSP_FM_DEVIATION_HZ`      | 75000.0 | Standard FM peak deviation (±75kHz). Used for proper audio level scaling            | Fixed for standard |

### Status and Buffer Management

| Constant                      | Value   | Purpose                                                                          | Configurable Range |
| ----------------------------- | ------- | -------------------------------------------------------------------------------- | ------------------ |
| `STATUS_UPDATE_INTERVAL_SEC`  | 2       | Console status update frequency. Balances user feedback vs. output spam          | 1-10 seconds       |
| `BUFFER_HIGH_THRESHOLD_PCT`   | 80.0    | Audio buffer high warning threshold. Prevents dropouts from buffer overflow      | 70-90%             |
| `BUFFER_LOW_THRESHOLD_PCT`    | 10.0    | Audio buffer low warning threshold. Indicates possible underrun                  | 5-20%              |
| `MIN_SAMPLES_FOR_LOW_WARNING` | 2500000 | Minimum samples before low buffer warnings. Prevents false alarms during startup | 1000000-5000000    |

### Performance Notes

- **Lower audio buffer sizes** reduce latency but increase CPU overhead and dropout risk
- **Higher SDR sample rates** improve image rejection but require more CPU processing
- **Larger ring buffers** provide better jitter tolerance but increase memory usage and latency
- **Decimation factor** must equal `SDR_SAMPLE_RATE / AUDIO_SAMPLE_RATE` for proper operation

### Regional Variations

- **Europe/Asia**: FM band 87.5-108 MHz, 50μs de-emphasis
- **North America**: FM band 88-108 MHz, 75μs de-emphasis
- **Japan**: FM band 76-95 MHz, 50μs de-emphasis

_To modify these constants, edit `src/common.h` and rebuild the project._

## Requirements

- ADALM-PLUTO SDR
- FM antenna (simple wire works for local stations)
- Audio output device

## Performance

- **Real-time processing**: 2.4 MSPS sustained
- **Audio latency**: ~50ms buffering
- **Memory efficient**: Pre-allocated buffers, no malloc in hot path
- **CPU optimized**: Explicit data flow, minimal overhead
