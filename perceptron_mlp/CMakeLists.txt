cmake_minimum_required(VERSION 3.10)
project(PerceptronMLP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Specify the source files
set(SOURCES
    src/main.cu
    src/layers/dense_layer.cu
    src/layers/activation.cu
    src/losses/softmax_loss.cu
)

# Include directories
include_directories(
    src
    src/layers
    src/losses
    src/utils
    src/tests
)

# Create the executable
add_executable(PerceptronMLP ${SOURCES})

# Link CUDA libraries
find_package(CUDA REQUIRED)
target_link_libraries(PerceptronMLP ${CUDA_LIBRARIES})