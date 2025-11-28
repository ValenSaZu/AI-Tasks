// filepath: c:\Users\INTEL\Documents\Universidad\2025-2\IA\AI-Tasks\AI-Tasks\perceptron_mlp\PerceptronMLP\src\losses\softmax_loss.cu

#include "softmax_loss.h"
#include <cuda_runtime.h>

__device__ float softmax(const float* logits, int numClasses) {
    float maxLogit = logits[0];
    for (int i = 1; i < numClasses; i++) {
        if (logits[i] > maxLogit) {
            maxLogit = logits[i];
        }
    }

    float sumExp = 0.0f;
    for (int i = 0; i < numClasses; i++) {
        sumExp += expf(logits[i] - maxLogit);
    }

    return sumExp;
}

__global__ void softmaxKernel(const float* logits, float* probabilities, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClasses) {
        float sumExp = softmax(logits, numClasses);
        probabilities[idx] = expf(logits[idx]) / sumExp;
    }
}

__global__ void softmaxLossKernel(const float* probabilities, const int* targets, float* loss, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClasses) {
        float targetProb = (targets[idx] == 1) ? 1.0f : 0.0f;
        loss[0] -= targetProb * logf(probabilities[idx]);
    }
}

void SoftmaxLoss::forward(const float* logits, const int* targets, float* loss, float* probabilities, int numClasses) {
    softmaxKernel<<<1, numClasses>>>(logits, probabilities, numClasses);
    cudaDeviceSynchronize();

    softmaxLossKernel<<<1, numClasses>>>(probabilities, targets, loss, numClasses);
    cudaDeviceSynchronize();
}

__global__ void softmaxLossGradientKernel(const float* probabilities, const int* targets, float* gradients, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numClasses) {
        gradients[idx] = probabilities[idx] - (targets[idx] == 1 ? 1.0f : 0.0f);
    }
}

void SoftmaxLoss::backward(const float* probabilities, const int* targets, float* gradients, int numClasses) {
    softmaxLossGradientKernel<<<1, numClasses>>>(probabilities, targets, gradients, numClasses);
    cudaDeviceSynchronize();
}