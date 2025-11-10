#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>
#include <iostream>
#include <cmath> // para softmax

using namespace std;

struct Edge {
    int destination; // índice de la neurona de destino
    float weight; // peso de la neurona

    Edge(int dest) {
        destination = dest;
        weight = 0.0f;
    }
};

struct Neuron {
    int id;
    float inputValue;
    vector<Edge> neighbors;
    float outputValue;

    Neuron(int idx = -1){
        id = idx;
        inputValue = 0.0f;
        outputValue = 0.0f;
    }

    void addEdge(int dest) {
        neighbors.emplace_back(dest);
}
};

class Perceptron {
    float learningRate;
    int numOutputs;

public:
    vector<Neuron> neuronsInputLayer; // 3072 + 1 (bias)
    vector<float> outputValues; // 10 salidas, solo obtenemos el valor

    Perceptron(int numInputs, int nOutputs, float lrate = 0.1f){
        learningRate = lrate;
        numOutputs = nOutputs;
        outputValues = vector<float>(numOutputs, 0.0f);
        constructPerceptron(numInputs);
        initializeWeights();
    }

    
    void constructPerceptron(int numInputs) {
        // Capa de entrada + bias
        for (int i = 0; i < numInputs + 1; i++) {
            neuronsInputLayer.emplace_back(i);
        }
        // Inicializar valores de entrada
        neuronsInputLayer[0].inputValue = 1.0f; // bias

        // Conectar cada neurona de entrada con cada neurona de salida
        for (int i = 0; i < neuronsInputLayer.size(); i++) {
            for (int j = 0; j < numOutputs; j++) {
                neuronsInputLayer[i].addEdge(j);
            }
        }
    }

    void initializeWeights() {
        for (auto &neuron : neuronsInputLayer) {
            for (auto &edge : neuron.neighbors) {
                // Pesos pequeños aleatorios
                edge.weight = static_cast<float>(rand()) / RAND_MAX * 0.01f;
            }
        }
    }

    // Calcula las salidas, un vector de probabilidades, una por clase
    vector<float> calculateOutputs() {
        //suma de cada entrada para cada salida
        vector<float> sums(numOutputs, 0.0f);

        // input1 * weight1 + input2 * weight2 + ...
        for (auto &neuron : neuronsInputLayer) {
            for (auto &edge : neuron.neighbors) {
                sums[edge.destination] += neuron.inputValue * edge.weight;
            }
        }

        // Activación tipo softmax para clasificación multiclase
        // Softmax: Convierte los valores en probabilidades que suman 1
        float sumExp = 0.0f;
        for (int i = 0; i < numOutputs; i++) {
            sumExp += exp(sums[i]);
        }

        for (int i = 0; i < numOutputs; i++) {
            outputValues[i] = exp(sums[i]) / sumExp;
        }

        return outputValues;
    }

    void setInputs(const vector<float> &inputs) {
        for (int i = 0; i < inputs.size(); i++) {
            neuronsInputLayer[i + 1].inputValue = inputs[i];
        }
    }

    // Entrenamiento normal de la teoria del perceptrón
    void train(const vector<vector<float>> trainData, const vector<int> labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalError = 0.0f;

            for (int n = 0; n < trainData.size(); n++) {
                setInputs(trainData[n]);
                vector<float> outputs = calculateOutputs();

                vector<float> expected(numOutputs, 0.0f);
                expected[labels[n]] = 1.0f;

                // actualiza pesos
                for (auto &neuron : neuronsInputLayer) {
                    for (auto &edge : neuron.neighbors) {
                        float error = expected[edge.destination] - outputs[edge.destination];
                        //fabs convierte a positivo en caso de error negativo
                        totalError += fabs(error);
                        edge.weight += learningRate * error * neuron.inputValue;
                    }
                }
            }

            cout << "Época " << epoch + 1 << " completada. Error total: " << totalError << endl;
        }
    }

    //se usa para la fase de prueba, devuelve la clase predicha
    int predict(const vector<float> &inputs) {
        setInputs(inputs);
        vector<float> outputs = calculateOutputs();
        //la prediccion, mayor probabilidad
        // retorna la distancia del inicio de las salidas al máximo elemento
        // o sea el indice en el vector, por lo tanto la clase
        return distance(outputs.begin(), max_element(outputs.begin(), outputs.end()));
    }
};

#endif
