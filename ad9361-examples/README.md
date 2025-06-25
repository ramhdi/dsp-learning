# AD9361 IIO Streaming Example

A real-time SDR streaming application demonstrating full-duplex signal processing with the ADALM-PLUTO using libiio.

## Overview

This example implements a complete software-defined radio pipeline that:
- Configures AD9361 transceiver for 2.5 GHz operation
- Simultaneously receives and transmits I/Q samples
- Demonstrates real-time signal processing (I/Q swap on RX path)
- Uses 1 MiS buffers for continuous streaming at 2.5 MSPS

## Prerequisites

- **Hardware**: ADALM-PLUTO SDR with USB drivers installed
- **Software**: MSYS2 with UCRT64 environment
- **Libraries**: libiio (`pacman -S mingw-w64-ucrt-x86_64-libiio`)

## Build Instructions

```bash
# Create project directory
mkdir ad9361-examples && cd ad9361-examples

# Copy example source
cp path/to/libiio/examples/ad9361-iiostream.c .

# Create CMakeLists.txt (see below)

# Build
cmake -G Ninja .
ninja
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(ad9361_examples)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(IIO REQUIRED libiio)

# Add compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

# Include directories
include_directories(${IIO_INCLUDE_DIRS})

# Add executable
add_executable(ad9361_stream ad9361-iiostream.c)

# Link libraries
target_link_libraries(ad9361_stream ${IIO_LIBRARIES})

# Compile definitions
target_compile_definitions(ad9361_stream PRIVATE ${IIO_CFLAGS_OTHER})
```

## Usage

```bash
# Connect to PLUTO via network (default)
./ad9361_stream.exe ip:192.168.2.1

# Auto-detect connection
./ad9361_stream.exe

# Connect via USB (if supported)
./ad9361_stream.exe usb:
```

## Signal Configuration

- **RX**: 2.5 GHz center frequency, 2 MHz bandwidth, 2.5 MSPS
- **TX**: 2.5 GHz center frequency, 1.5 MHz bandwidth, 2.5 MSPS
- **Processing**: I/Q swap on received samples
- **Output**: Transmits zeros (configurable in code)

## Code Structure

| Function | Purpose |
|----------|---------|
| `cfg_ad9361_streaming_ch()` | Configure RF parameters (frequency, bandwidth, sample rate) |
| `get_ad9361_stream_dev()` | Acquire streaming devices for RX/TX |
| `main()` loop | Real-time buffer management and signal processing |

## Real-Time Processing Loop

1. **Push TX buffer** → Transmit samples to antenna
2. **Refill RX buffer** → Receive samples from antenna
3. **Process RX samples** → Apply DSP algorithms (I/Q swap demo)
4. **Generate TX samples** → Create waveform for transmission
5. **Repeat** at sample rate (2.5 MSPS)

## Customization

Replace the example processing sections with your own DSP algorithms:

```c
// RX Processing (lines 286-294)
for (p_dat = (char *) iio_buffer_first(rxbuf, rx0_i); p_dat < p_end; p_dat += p_inc) {
    // Your RX signal processing here
}

// TX Generation (lines 296-305) 
for (p_dat = (char *) iio_buffer_first(txbuf, tx0_i); p_dat < p_end; p_dat += p_inc) {
    // Your TX waveform generation here
}
```

## Troubleshooting

- **"No such file"**: Check executable has `.exe` extension
- **"No devices found"**: Verify PLUTO connection with `ping 192.168.2.1`
- **Build errors**: Ensure libiio package installed in MSYS2
- **Performance issues**: Monitor buffer underruns in console output

## Sample Rate Math

At 2.5 MSPS with 1 MiS buffers:
- **Buffer duration**: 400ms per buffer
- **Processing deadline**: 400ns per sample
- **Memory bandwidth**: ~20 MB/s for I/Q int16 samples