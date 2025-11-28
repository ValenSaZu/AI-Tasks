// filepath: c:\Users\INTEL\Documents\Universidad\2025-2\IA\AI-Tasks\AI-Tasks\perceptron_mlp\PerceptronMLP\src\layers\activation.cu

#include "activation.h"
#include <cuda_runtime.h>
#include <cmath>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoidKernel(const float* inputs, float* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outputs[idx] = sigmoid(inputs[idx]);
    }
}

__device__ void softmax(const float* inputs, float* outputs, int size) {
    float maxVal = inputs[0];
    for (int i = 1; i < size; i++) {
        if (inputs[i] > maxVal) {
            maxVal = inputs[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        outputs[i] = expf(inputs[i] - maxVal);
        sum += outputs[i];
    }

    for (int i = 0; i < size; i++) {
        outputs[i] /= sum;
    }
}

__global__ void softmaxKernel(const float* inputs, float* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        softmax(inputs, outputs, size);
    }
}