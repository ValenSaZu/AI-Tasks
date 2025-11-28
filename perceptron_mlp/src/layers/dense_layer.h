#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <vector>
#include <cuda_runtime.h>
#include "activation.h"

class DenseLayer {
private:
    int inputSize;
    int outputSize;
    float* d_weights;
    float* d_biases;

public:
    DenseLayer(int inputSize, int outputSize);
    ~DenseLayer();

    void forward(const float* d_input, float* d_output, int batchSize);
    void backward(const float* d_input, const float* d_output, const float* d_gradOutput, float* d_gradInput, int batchSize);
    void updateWeights(float learningRate, int batchSize);
};

#endif