cmake_minimum_required(VERSION 3.15)
project(fm_demodulator_soapy LANGUAGES CXX)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(PkgConfig REQUIRED)

# Find SoapySDR
pkg_check_modules(SOAPYSDR REQUIRED SoapySDR)

# Find PortAudio  
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Source files
set(SOURCES
    src/main.cpp
    src/dsp.cpp
    src/sdr_interface.cpp
    src/audio_interface.cpp
)

# Headers (for IDE support)
set(HEADERS
    src/types.hpp
    src/dsp.hpp
    src/sdr_interface.hpp
    src/audio_interface.hpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    src
    ${SOAPYSDR_INCLUDE_DIRS}
    ${PORTAUDIO_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${SOAPYSDR_CFLAGS_OTHER}
    ${PORTAUDIO_CFLAGS_OTHER}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${SOAPYSDR_LIBRARIES}
    ${PORTAUDIO_LIBRARIES}
)

# Link directories (if needed)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${SOAPYSDR_LIBRARY_DIRS}
    ${PORTAUDIO_LIBRARY_DIRS}
)

# Windows-specific settings
if(WIN32)
    # Ensure we link against Windows libraries that PortAudio might need
    target_link_libraries(${PROJECT_NAME} winmm)
endif()

# Install target (optional)
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)