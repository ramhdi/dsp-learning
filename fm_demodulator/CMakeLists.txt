cmake_minimum_required(VERSION 3.15)
project(fm_demodulator)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include our headers
include_directories(include src)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(IIO REQUIRED libiio)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -g")

# Source files
set(SOURCES
    src/main.c
    src/processing.c
    src/io.c
)

# Create executable
add_executable(fm_demodulator ${SOURCES})

# Include directories for dependencies
target_include_directories(fm_demodulator PRIVATE
    ${IIO_INCLUDE_DIRS}
    ${PORTAUDIO_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(fm_demodulator
    ${IIO_LIBRARIES}
    ${PORTAUDIO_LIBRARIES}
    m  # Math library for atan2f, etc.
)

# Compile definitions
target_compile_definitions(fm_demodulator PRIVATE
    ${IIO_CFLAGS_OTHER}
    ${PORTAUDIO_CFLAGS_OTHER}
)