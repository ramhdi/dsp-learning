# ADALM-PLUTO FM Demodulator

Real-time FM radio demodulator using ADALM-PLUTO SDR with audio playback.

## Features

- Real-time FM demodulation using phase discriminator
- Audio playback via PortAudio
- Configurable FM station tuning
- Cross-platform support

## Project Structure

```
fm_demodulator/
├── CMakeLists.txt
├── include/
│   ├── common.h
│   ├── audio.h
│   ├── pluto.h
│   └── demodulator.h
└── src/
    ├── main.c
    ├── audio.c
    ├── pluto.c
    └── demodulator.c
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

Edit `include/common.h` to change default settings:

- `DEFAULT_FM_FREQ` - Default station frequency
- `FM_SAMPLE_RATE` - RF sample rate (higher = better quality)
- `MAX_AUDIO_AMPLITUDE` - Volume control

## Requirements

- ADALM-PLUTO SDR
- FM antenna (simple wire works for local stations)
- Audio output device
