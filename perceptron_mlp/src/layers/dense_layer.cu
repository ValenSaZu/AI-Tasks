// filepath: c:\Users\INTEL\Documents\Universidad\2025-2\IA\AI-Tasks\AI-Tasks\perceptron_mlp\PerceptronMLP\src\layers\dense_layer.cu

#include "dense_layer.h"
#include "activation.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void forwardKernel(const float* inputs, const float* weights, float* outputs, int numInputs, int numOutputs) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numOutputs) {
        float sum = 0.0f;
        for (int i = 0; i < numInputs; i++) {
            sum += inputs[i] * weights[row * numInputs + i];
        }
        outputs[row] = sum;
    }
}

__global__ void backwardKernel(const float* inputs, const float* outputs, const float* gradOutputs, float* gradInputs, float* gradWeights, int numInputs, int numOutputs) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numInputs) {
        float grad = 0.0f;
        for (int j = 0; j < numOutputs; j++) {
            grad += gradOutputs[j] * inputs[row];
        }
        gradInputs[row] = grad;
    }
}

__global__ void updateWeightsKernel(float* weights, const float* gradWeights, float learningRate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learningRate * gradWeights[idx];
    }
}

DenseLayer::DenseLayer(int inputSize, int outputSize) 
    : inputSize(inputSize), outputSize(outputSize) {
    cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
    cudaMalloc(&d_gradWeights, inputSize * outputSize * sizeof(float));
    cudaMalloc(&d_outputs, outputSize * sizeof(float));
}

DenseLayer::~DenseLayer() {
    cudaFree(d_weights);
    cudaFree(d_gradWeights);
    cudaFree(d_outputs);
}

void DenseLayer::forward(const float* inputs, float* outputs) {
    dim3 block(256);
    dim3 grid((outputSize + block.x - 1) / block.x);
    forwardKernel<<<grid, block>>>(inputs, d_weights, d_outputs, inputSize, outputSize);
    cudaMemcpy(outputs, d_outputs, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void DenseLayer::backward(const float* inputs, const float* gradOutputs, float* gradInputs) {
    dim3 block(256);
    dim3 grid((inputSize + block.x - 1) / block.x);
    backwardKernel<<<grid, block>>>(inputs, d_outputs, gradOutputs, gradInputs, d_gradWeights, inputSize, outputSize);
}

void DenseLayer::updateWeights(float learningRate) {
    dim3 block(256);
    dim3 grid((inputSize * outputSize + block.x - 1) / block.x);
    updateWeightsKernel<<<grid, block>>>(d_weights, d_gradWeights, learningRate, inputSize * outputSize);
}