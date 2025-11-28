//
//  Created by Amara Barrera on 2/11/25.
//  Modified by Maria Calle on 15/11/25.
//  Made with the original perceptron as base, but adapted to be multilayer perceptron (MLP) using CUDA for GPU acceleration.
//

#include "cifar10_reader.h"
#include "perceptron_mlp.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {
    cout << "INICIANDO ENTRENAMIENTO DEL PERCEPTRÓN EN GPU" << endl;

    string dataPath = "/content/cifar-10-binary";
    int batchSize = 256;
    int inputDim = 3072;
    int numClasses = 10;
    int epochs = 1000;
    float learningRate = 0.01f;

    DataLoader loader(dataPath, batchSize, inputDim, numClasses);

    cout << "Cargando 5 archivos de entrenamiento" << endl;
    for (int i = 1; i <= 5; i++) {
        string fname = dataPath + "/data_batch_" + to_string(i) + ".bin";
        loader.readBinaryFile(fname, loader.trainingImages, loader.trainingLabels);
        cout << "  Cargado data_batch_" << i << ".bin → Total: " << loader.trainingImages.size() << endl;
    }
    cout << "Total entrenamiento: " << loader.trainingImages.size() << " imágenes." << endl;

    string test_fname = dataPath + "/test_batch.bin";
    loader.readBinaryFile(test_fname, loader.testImages, loader.testLabels);
    cout << "Cargado test_batch.bin: " << loader.testImages.size() << " imágenes." << endl;

    auto& trainImages = loader.trainingImages;
    auto& trainLabels = loader.trainingLabels;
    int totalTrain = int(trainImages.size());

    int numBatchesPerEpoch = (totalTrain + batchSize - 1) / batchSize;
    cout << "Configuración: " << numBatchesPerEpoch << " batches por época." << endl;

    Perceptron perceptron(inputDim, numClasses, learningRate);
    random_device rd;
    mt19937 g(rd());

    cout << fixed << setprecision(2);
    cout << "INICIANDO ENTRENAMIENTO" << endl;

    auto start_time = high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        vector<int> indices(totalTrain);
        for (int i = 0; i < totalTrain; i++) indices[i] = i;
        shuffle(indices.begin(), indices.end(), g);

        float epochLoss = 0.0f, epochAcc = 0.0f;

        for (int b = 0; b < numBatchesPerEpoch; b++) {
            vector<vector<float>> batchImages;
            vector<int> batchLabels;
            int start = b * batchSize;

            int currentBatchSize = min(batchSize, totalTrain - start);

            for (int i = 0; i < currentBatchSize; i++) {
                int idx = indices[start + i];
                batchImages.push_back(trainImages[idx]);
                batchLabels.push_back(trainLabels[idx]);
            }

            float batchLoss, batchAcc;
            perceptron.trainBatch(batchImages, batchLabels, batchLoss, batchAcc);

            epochLoss += batchLoss;
            epochAcc += batchAcc;

            cout << "\rÉpoca " << (epoch+1) << " | Batch " << (b+1) << "/" << numBatchesPerEpoch
                 << " | Loss: " << batchLoss << " | Acc: " << batchAcc << "%" << flush;
        }

        epochLoss /= numBatchesPerEpoch;
        epochAcc /= numBatchesPerEpoch;

        cout << "\n>>> ÉPOCA " << (epoch+1) << " COMPLETADA: "
             << "Loss: " << epochLoss << " | Acc: " << epochAcc << "%" << endl;
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end_time - start_time);
    cout << "ENTRENAMIENTO COMPLETADO EN " << duration.count() << " SEGUNDOS." << endl;

    cout << "\nPRUEBA EN 10 IMÁGENES" << endl;
    auto& testImages = loader.testImages;
    auto& testLabels = loader.testLabels;
    int correct = 0;
    for (int i = 0; i < 10; i++) {
        int pred = perceptron.predict(testImages[i]);
        bool ok = (pred == testLabels[i]);
        if (ok) correct++;
        cout << "Img " << i << " → Pred: " << pred << " | Real: " << testLabels[i]
             << (ok ? " [OK]" : " [ERROR]") << endl;
    }
    cout << "Accuracy en 10 imágenes: " << (100.0 * correct / 10) << "%" << endl;

    cout << "TODO COMPLETADO" << endl;
    return 0;
}