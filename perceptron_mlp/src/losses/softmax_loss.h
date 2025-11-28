#ifndef SOFTMAX_LOSS_HPP
#define SOFTMAX_LOSS_HPP

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "activation.h"

class SoftmaxLoss {
public:
    // Constructor
    SoftmaxLoss() {}

    // Compute the softmax loss and gradients
    __host__ __device__ void computeLossAndGradients(const float* outputs, const int* targets, int numSamples, int numClasses, float& loss, float* gradients) {
        // Compute softmax
        float* softmaxOutputs = new float[numSamples * numClasses];
        softmax(outputs, softmaxOutputs, numSamples, numClasses);

        // Compute loss and gradients
        loss = 0.0f;
        for (int i = 0; i < numSamples; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                if (j == targets[i]) {
                    loss -= log(softmaxOutputs[i * numClasses + j]);
                    gradients[i * numClasses + j] = softmaxOutputs[i * numClasses + j] - 1.0f; // Derivative for correct class
                } else {
                    gradients[i * numClasses + j] = softmaxOutputs[i * numClasses + j]; // Derivative for incorrect classes
                }
            }
        }
        delete[] softmaxOutputs;
        loss /= numSamples; // Average loss
    }
};

#endif