cmake_minimum_required(VERSION 3.15)
project(ad9361_examples)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages (only libiio needed for this example)
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
