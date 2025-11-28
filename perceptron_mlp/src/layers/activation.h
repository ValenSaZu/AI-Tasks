#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Sigmoid activation function for a batch of inputs
__global__ void sigmoidActivationKernel(const float* inputs, float* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outputs[idx] = sigmoid(inputs[idx]);
    }
}

// Softmax activation function
__device__ void softmax(const float* inputs, float* outputs, int size) {
    float maxVal = inputs[0];
    for (int i = 1; i < size; ++i) {
        if (inputs[i] > maxVal) {
            maxVal = inputs[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        outputs[i] = expf(inputs[i] - maxVal);
        sum += outputs[i];
    }

    for (int i = 0; i < size; ++i) {
        outputs[i] /= sum;
    }
}

// Softmax activation function for a batch of inputs
__global__ void softmaxActivationKernel(const float* inputs, float* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        softmax(inputs + idx * size, outputs + idx * size, size);
    }
}

#endif // ACTIVATION_H