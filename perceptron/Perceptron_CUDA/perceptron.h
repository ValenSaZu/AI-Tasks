//
//  perceptron.h
//  Perceptron
//
//  Created by Amara Barrera on 2/11/25.
//

//Holaaa, ya los principales cambios son:
    //1. Con el código en c++ es que se quitó la estructura grafo que se había implementado para el perceptron (edge y neuron). Ponerlo como matriz es más fácil y rápido para implementarlo en cuda
    //2. La función calculteOutputs, setInputs y train dentro de la clase perceptron del c++, se dividen en forwardKernel y backwardKernel

#ifndef PERCEPTRON_CUDA_HPP
#define PERCEPTRON_CUDA_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
using namespace std;

//Macro para chequear errores
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA ERROR: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

//En c++ había teníamos la función "Train" entonces la dividimos en kernels de forward y backward
//Kernel de forward
__global__ void forwardKernel(const float* inputs, const float* weights, float* outputs, int N, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; //Indice global de CUDA
    int col = blockIdx.y * blockDim.y + threadIdx.y; //Indice global de CUDA
    if (row >= N || col >= K)
        return;

    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += inputs[row * M + i] * weights[i * K + col];
    }
    outputs[row * K + col] = sum;
}

//Kernel de backward
__global__ void backwardKernel(float* grad_weights, const float* inputs, const float* outputs, const float* targets, int N, int M, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= K)
        return;

    float grad = 0.0f;
    for (int n = 0; n < N; n++) {
        float act = outputs[n * K + j];
        float err = act - targets[n * K + j];
        grad += err * inputs[n * M + i];
    }
    grad_weights[i * K + j] = grad / N;
}

//Kernel para pesos
__global__ void updateWeightsKernel(float* weights, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    weights[idx] -= lr * grad[idx];
}

//Clase perceptron
class Perceptron {
private:
    float learningRate;
    int numInputs;
    int numOutputs;
    int totalWeights;
    int batch_size = 256;

    float* d_weights; //usamos una matriz de pesos
    float* d_inputs;
    float* d_outputs;
    float* d_targets;
    float* d_grad;

public:
    Perceptron(int nInputs = 3072, int nOutputs = 10, float lr = 0.1f)
        : numInputs(nInputs), numOutputs(nOutputs), learningRate(lr) {
        totalWeights = (nInputs + 1) * nOutputs;

        // Reservando memoria en cuda
            // Usamos d_ para definir que es del device del GPU
        CUDA_CHECK(cudaMalloc(&d_weights, totalWeights * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_inputs, batch_size * (nInputs + 1) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputs, batch_size * nOutputs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, batch_size * nOutputs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, totalWeights * sizeof(float)));

        // Inicializar pesos
            // Creamos el vector h_weights que vamos a usar en el CPU: guarda pesos temporalmente
        vector<float> h_weights(totalWeights);
        // Generamos pesos aleatorios con mt19937
        random_device rd;
        mt19937 gen(rd());

        // En una de las pruebas, los resultados del forward se volvían dms grandes, entonces vamos a ponerles un límite con la fórmula 1.0f/sqrt(numInputs)
        // Los va a inicializar con esta forma [-limit, limit]
        // Así se previene que las salidas del forward sean demasiado grandes al inicio y que de como salidas "nan"
        float limit = 1.0f / sqrtf(numInputs);
            // Usamos este objeto uniform_real_distribution que produce nums que siguen reglas específicas
        uniform_real_distribution<float> dis(-limit, limit); // usamos dis para que nos de un número entre los límites calculados
        
        // Aquí se llena el vector del CPU con números aleatorios
        for (int i = 0; i < totalWeights; i++)
            h_weights[i] = dis(gen);
        // Copiamos el vector del CPu al GPU
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), totalWeights * sizeof(float), cudaMemcpyHostToDevice)); // Pasamos datos de CPU a GPU con HostToDevice
    }

    // Destructor
    ~Perceptron() {
        cudaFree(d_weights); cudaFree(d_inputs); cudaFree(d_outputs);
        cudaFree(d_targets); cudaFree(d_grad);
    }

    // El trainBatch
    void trainBatch(const vector<vector<float>>& images, const vector<int>& labels,
                    float& batchLoss, float& batchAcc) {
        int N = images.size(); // num de imagenes
        int M = numInputs + 1; // tamaño de una sola imagen + 1 del bias

        vector<float> h_inputs(N * M, 0.0f); // inicializando con 0.0f
        // Este es el que va a cambiar a 1 cuando la clase sea correcta
        vector<float> h_targets(N * numOutputs, 0.0f); // inicializando con 0.0f

        // Donde N = 256
        for (int n = 0; n < N; n++) {
            h_inputs[n * M] = 1.0f;  // bias
            for (int i = 0; i < numInputs; i++) {
                // Normalizar los datos haciendo, cuando no se normalizaba explotaba:
                // dividir el valor del píxel (0-255) por 255.0 para que esté en el rango [0, 1]
                h_inputs[n * M + 1 + i] = images[n][i] / 255.0f; //n*M salta a la imagen correcta, + 1 salta el bias y coloca el pixel en i
                
            }
            h_targets[n * numOutputs + labels[n]] = 1.0f; //se ubica en el indice correcto y coloca 1.0f para indicar a qué clase pertence esa img
        }

        // Con cudaMemcpyHostToDevice pasamos los datos a GPU
        CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), N * numOutputs * sizeof(float), cudaMemcpyHostToDevice));

        // Define la cuadrícula para los threads de GPU - dim3 es un struct de CUDA (guarda 3 nums juntos) (configuración de ejecución en pocas palabras)
        dim3 block(16, 16); //tamaño de un solo bloque
        dim3 grid((N + 15)/16, (numOutputs + 15)/16); //tamaño de cuadricula total

        // Kernel Forward para llamada de lanzamiento del kernel
        forwardKernel<<<grid, block>>>(d_inputs, d_weights, d_outputs, N, M, numOutputs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Vector que guarda las puntuaciones de la red
        vector<float> h_outputs(N * numOutputs);
        CUDA_CHECK(cudaMemcpy(h_outputs.data(), d_outputs, N * numOutputs * sizeof(float), cudaMemcpyDeviceToHost));

        // Loss y Accuracy
        batchLoss = 0.0f;
        int correct = 0;
        for (int n = 0; n < N; n++) {
            //Calcula accuracy
            int pred = 0;
            for (int j = 1; j < numOutputs; j++) { //Itera sobre las otras 9 clases
                //Compara puntuación de clase j con la de la case pred (suponemos que es la mejor hasta ahora)
                if (h_outputs[n * numOutputs + j] > h_outputs[n * numOutputs + pred]) pred = j;
            }
            // Si es igual a la etiqueta acertó
            if (pred == labels[n])
                correct++;

            for (int j = 0; j < numOutputs; j++) {
                float err = h_outputs[n * numOutputs + j] - h_targets[n * numOutputs + j]; // calcula diferencia error
                batchLoss += err * err; // eleva error al cuadrado para penalizar errores grandes y hacerlo todo positivo
            }
        }
        batchLoss /= N; // calcula el promedio de loss
        batchAcc = 100.0f * correct / N; // calcula porcentaje de aciertos

        // Backward
        // Se pasan los datos al GPU
        CUDA_CHECK(cudaMemcpy(d_outputs, h_outputs.data(), N * numOutputs * sizeof(float), cudaMemcpyHostToDevice));
        
        //nuevo tamaño de cuadricula para Backward
        dim3 grid2((M + 15)/16, (numOutputs + 15)/16);
        // Llamada al kernel de backward con "blcok" que se definió antes con dim3
        backwardKernel<<<grid2, block>>>(d_grad, d_inputs, d_outputs, d_targets, N, M, numOutputs);
        
        // Llamada a la macro de cuda para verificar errores
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Llamada al kernel que actualiza datos
        updateWeightsKernel<<<(totalWeights + 255)/256, 256>>>(d_weights, d_grad, learningRate, totalWeights);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // se usa para la fase de prueba, devuelve la clase predicha
    int predict(const vector<float>& input) {
        vector<float> h_in(numInputs + 1, 0.0f);
        h_in[0] = 1.0f;
        for (int i = 0; i < numInputs; i++) {
            // Normalizamos de nuevo para que se quede en el rango y no explote
            h_in[1 + i] = input[i] / 255.0f;
        }

        CUDA_CHECK(cudaMemcpy(d_inputs, h_in.data(), (numInputs + 1) * sizeof(float), cudaMemcpyHostToDevice));
        dim3 block(16, 16); // tamaño de un solo bloque
        dim3 grid(1, 1); // tamaño de cuadricula total
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
