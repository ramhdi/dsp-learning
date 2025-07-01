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

Default settings in `src/common.h`:

- **Sample Rate**: 2.4 MSPS (PlutoSDR minimum requirement)
- **Audio Rate**: 48 kHz
- **Decimation**: 50x (2400 kHz → 48 kHz)
- **Audio Buffer**: 81920 samples (~1.7 second buffer)
- **Max Audio Amplitude**: 0.4 (volume control)

## Requirements

- ADALM-PLUTO SDR
- FM antenna (simple wire works for local stations)
- Audio output device

## Performance

- **Real-time processing**: 2.4 MSPS sustained
- **Audio latency**: ~50ms buffering
- **Memory efficient**: Pre-allocated buffers, no malloc in hot path
- **CPU optimized**: Explicit data flow, minimal overhead
