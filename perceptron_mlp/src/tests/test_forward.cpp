#include <gtest/gtest.h>
#include "../perceptron_mlp.h"
#include "../cifar10_reader.h"

class PerceptronMLPTest : public ::testing::Test {
protected:
    Perceptron* perceptron;
    DataLoader* dataLoader;

    void SetUp() override {
        // Initialize the Perceptron with input dimension and number of classes
        perceptron = new Perceptron(3072, 10, 0.01f);
        
        // Initialize DataLoader with a dummy path
        dataLoader = new DataLoader("dummy_path", 256, 3072, 10);
        
        // Load dummy data for testing
        dataLoader->loadTrainingData();
    }

    void TearDown() override {
        delete perceptron;
        delete dataLoader;
    }
};

TEST_F(PerceptronMLPTest, ForwardPassTest) {
    // Prepare a batch of images and labels
    vector<vector<float>> batchImages = dataLoader->trainingImages;
    vector<int> batchLabels = dataLoader->trainingLabels;

    float batchLoss, batchAcc;
    perceptron->trainBatch(batchImages, batchLabels, batchLoss, batchAcc);

    // Check if the loss and accuracy are within expected ranges
    EXPECT_GE(batchLoss, 0.0f);
    EXPECT_LE(batchAcc, 100.0f);
}

TEST_F(PerceptronMLPTest, PredictTest) {
    // Test prediction on a single image
    int pred = perceptron->predict(dataLoader->trainingImages[0]);
    EXPECT_GE(pred, 0);
    EXPECT_LT(pred, 10); // Assuming 10 classes
}