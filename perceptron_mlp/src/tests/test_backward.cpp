#include "perceptron_mlp.h"
#include "cifar10_reader.h"
#include <cassert>
#include <iostream>
#include <vector>

void testBackward() {
    // Initialize parameters
    int inputDim = 3072; // CIFAR-10 image size
    int numClasses = 10; // Number of classes
    float learningRate = 0.01f;

    // Create a Perceptron instance
    Perceptron perceptron(inputDim, numClasses, learningRate);

    // Create dummy data for testing
    std::vector<std::vector<float>> images(256, std::vector<float>(inputDim, 0.5f)); // 256 images with all pixel values set to 0.5
    std::vector<int> labels(256, 0); // All labels set to class 0

    // Variables to hold loss and accuracy
    float batchLoss, batchAcc;

    // Train the perceptron with the dummy data
    perceptron.trainBatch(images, labels, batchLoss, batchAcc);

    // Check if the loss and accuracy are within expected ranges
    assert(batchLoss >= 0.0f);
    assert(batchAcc >= 0.0f && batchAcc <= 100.0f);

    std::cout << "Backward test passed: Loss = " << batchLoss << ", Accuracy = " << batchAcc << "%" << std::endl;
}

int main() {
    testBackward();
    return 0;
}