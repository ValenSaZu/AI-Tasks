#include "cifar10_reader.h"
#include "perceptron.h"
#include <iostream>

using namespace std;

int main() {

    // Configuración
    string dataPath = "../data/cifar-10-binary/cifar-10-batches-bin";
    int batchSize = 256;
    int inputDim = 3072; // 32x32x3
    int numClasses = 10;
    int epochs = 5;
    float learningRate = 0.01f;
    int numBatchesPerEpoch = 200; // cuántos batches usar por época

    // Cargar datos
    DataLoader loader(dataPath, batchSize, inputDim, numClasses);
    loader.loadTrainingData();

    // Crear perceptrón
    Perceptron perceptron(inputDim, numClasses, learningRate);

    // Entrenamiento sobre varios batches
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epochError = 0.0f;

        for (int b = 0; b < numBatchesPerEpoch; b++) {
            Batch batch = loader.getBatch();

            vector<vector<float>> inputs;
            vector<int> labels;

            for (int i = 0; i < batch.labels.size(); i++) {
                vector<float> img(batch.images.begin() + i * inputDim, batch.images.begin() + (i + 1) * inputDim);
                inputs.push_back(img);
                labels.push_back(batch.labels[i]);
            }

            // Entrenar con el batch actual
            perceptron.train(inputs, labels, 1); // 1 epoch por batch
        }

        cout << "Época " << epoch + 1 << " completada." << endl;
    }

    // testear el perceptrón
    loader.loadTestData();
    // Probar sobre las primeras 10 imágenes del set de test
    for (int i = 0; i < 10; i++) {
        int pred = perceptron.predict(loader.getTestImages()[i]);
        cout << "Imagen " << i 
            << ", Clase predicha: " << pred 
            << ", Clase real: " << loader.getTestLabels()[i] << endl;
    }

    return 0;
}
