#ifndef PERCEPTRON_MLP_HPP
#define PERCEPTRON_MLP_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include "activation.h"
#include "softmax_loss.h"
using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA ERROR: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

class Perceptron {
private:
    float learningRate;
    int numInputs;
    int numOutputs;
    int totalWeights;
    int batch_size = 256;

    float* d_weights;
    float* d_inputs;
    float* d_outputs;
    float* d_targets;
    float* d_grad;

public:
    Perceptron(int nInputs = 3072, int nOutputs = 10, float lr = 0.1f)
        : numInputs(nInputs), numOutputs(nOutputs), learningRate(lr) {
        totalWeights = (nInputs + 1) * nOutputs;

        CUDA_CHECK(cudaMalloc(&d_weights, totalWeights * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inputs, batch_size * (nInputs + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs, batch_size * nOutputs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, batch_size * nOutputs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, totalWeights * sizeof(float)));

        vector<float> h_weights(totalWeights);
        random_device rd;
        mt19937 gen(rd());
        float limit = 1.0f / sqrtf(numInputs);
        uniform_real_distribution<float> dis(-limit, limit);
        
        for (int i = 0; i < totalWeights; i++)
            h_weights[i] = dis(gen);
        
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), totalWeights * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~Perceptron() {
        cudaFree(d_weights); cudaFree(d_inputs); cudaFree(d_outputs);
        cudaFree(d_targets); cudaFree(d_grad);
    }

    void trainBatch(const vector<vector<float>>& images, const vector<int>& labels,
                    float& batchLoss, float& batchAcc) {
        int N = images.size();
        int M = numInputs + 1;

        vector<float> h_inputs(N * M, 0.0f);
        vector<float> h_targets(N * numOutputs, 0.0f);

        for (int n = 0; n < N; n++) {
            h_inputs[n * M] = 1.0f;
            for (int i = 0; i < numInputs; i++) {
                h_inputs[n * M + 1 + i] = images[n][i] / 255.0f;
            }
            h_targets[n * numOutputs + labels[n]] = 1.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), N * numOutputs * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N + 15)/16, (numOutputs + 15)/16);

        forwardKernel<<<grid, block>>>(d_inputs, d_weights, d_outputs, N, M, numOutputs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<float> h_outputs(N * numOutputs);
        CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_outputs, N * numOutputs * sizeof(float), cudaMemcpyDeviceToHost));

        batchLoss = 0.0f;
        int correct = 0;
        for (int n = 0; n < N; n++) {
            int pred = 0;
            for (int j = 1; j < numOutputs; j++) {
                if (h_outputs[n * numOutputs + j] > h_outputs[n * numOutputs + pred]) pred = j;
            }
            if (pred == labels[n]) correct++;

            for (int j = 0; j < numOutputs; j++) {
                float err = h_outputs[n * numOutputs + j] - h_targets[n * numOutputs + j];
                batchLoss += err * err;
            }
        }
        batchLoss /= N;
        batchAcc = 100.0f * correct / N;

        CUDA_CHECK(cudaMemcpy(d_outputs, h_outputs.data(), N * numOutputs * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 grid2((M + 15)/16, (numOutputs + 15)/16);
        backwardKernel<<<grid2, block>>>(d_grad, d_inputs, d_outputs, d_targets, N, M, numOutputs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        updateWeightsKernel<<<(totalWeights + 255)/256, 256>>>(d_weights, d_grad, learningRate, totalWeights);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int predict(const vector<float>& input) {
        vector<float> h_in(numInputs + 1, 0.0f);
        h_in[0] = 1.0f;
        for (int i = 0; i < numInputs; i++) {
            h_in[1 + i] = input[i] / 255.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_inputs, h_in.data(), (numInputs + 1) * sizeof(float), cudaMemcpyHostToDevice));
        dim3 block(16, 16);
        dim3 grid(1, 1);
        forwardKernel<<<grid, block>>>(d_inputs, d_weights, d_outputs, 1, numInputs + 1, numOutputs);
        CUDA_CHECK(cudaDeviceSynchronize());

        vector<float> h_out(numOutputs);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_outputs, numOutputs * sizeof(float), cudaMemcpyDeviceToHost));

        int pred = 0;
        float max_val = h_out[0];
        for (int j = 1; j < numOutputs; j++) {
            if (h_out[j] > max_val) {
                max_val = h_out[j];
                pred = j;
            }
        }
        return pred;
    }
};

#endif