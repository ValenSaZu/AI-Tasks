// codigo basado en: https://github.com/wichtounet/cifar-10/blob/master/include/cifar/cifar10_reader.hpp
#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cassert>
#include <memory>
#include <algorithm>

using namespace std;

struct Batch {
    vector<float> images; // tamaño = batch_size × input_dim
    vector<int>   labels; // tamaño = batch_size
};

class DataLoader {
public:
    DataLoader(const string& path, int batchSize, int inputDim, int numClasses)
        : data_path(path),
          batch_size(batchSize),
          input_dim(inputDim),
          num_classes(numClasses),
          rng(random_device{}())
    {}

    // Carga todos los datos de entrenamiento
    void loadTrainingData() {
        for (int i = 1; i <= 5; ++i) {
            string fname = data_path + "/data_batch_" + to_string(i) + ".bin";
            readBinaryFile(fname, trainingImages, trainingLabels);
        }
    }

    // Obtiene un batch aleatorio del conjunto cargado
    Batch getBatch() {
        Batch batch;
        if(trainingImages.size() == trainingLabels.size()){
            // Reservar espacio
            batch.images.resize(batch_size * input_dim);
            batch.labels.resize(batch_size);

            // Seleccionar muestras aleatorias entre 0 y el numero de imagenes cargadas
            // util para que todas tengan la misma probabilidad de ser elegidas
            uniform_int_distribution<int> dist(0, static_cast<int>(trainingImages.size()) - 1);

            for (int i = 0; i < batch_size; ++i) {
                // Aqui se usa la distribucion que se definio antes, obtiene un numero aleatorio en el rango
                int idx = dist(rng);
                // referencia a la imagen seleccionada
                // no es necesaria pero hacemos la referencia para no hacer la llamada cada vez al indice
                const vector<float>& src = trainingImages[idx];
                // copia la imagen en el batch, tomando en cuenta que numero de imagen es a partir de la entrada
                copy(src.begin(), src.end(),
                    batch.images.begin() + i * input_dim);
                //copia la etiqueta
                batch.labels[i] = trainingLabels[idx];
            }
        } else {
            cout << "Error, no se cargo correctamente los datos" << endl;
        }
        return batch;
    }

    void loadTestData() {
        string fname = data_path + "/test_batch.bin";
        readBinaryFile(fname, testImages, testLabels);
    }

    const vector<vector<float>>& getTestImages() const {
        return testImages;
    }
    const vector<int>& getTestLabels() const {
        return testLabels;
    }

public:
    string data_path;
    int batch_size;
    int input_dim;
    int num_classes;

    vector< vector<float> > trainingImages;
    vector< int > trainingLabels;

    vector<vector<float>> testImages;
    vector<int> testLabels;

    mt19937 rng;

    // Lectura de archivo CIFAR-10
    // Se agregan parámetros para decidir dónde guardar las imágenes y etiquetas
    void readBinaryFile(const string& filename, vector<vector<float>>& images, vector<int>& labels) {
        // abrimos el archivo para lectura, en modo binario y 
        // nos posicionamos al final para obtener el tamaño (luego se usa en tellg)
        ifstream file(filename, ios::in | ios::binary | ios::ate);
        if (!file) {
            cout << "Error con la imagen " << filename << endl;
            return;
        }

        // tellg basicamente obtiene la posicion del get pointer, que segun arriba, esta al final, asi que es el tamaño
        streampos file_size = file.tellg();
        // mueve el puntero al inicio
        file.seekg(0, ios::beg);

        // Cuanto mide un registro
        const int record_bytes = 1 + input_dim;
        // cuantos registros hay en el archivo
        size_t num_records = static_cast<size_t>(file_size) / record_bytes;

        // buffer para almacenar todo el archivo
        unique_ptr<char[]> buffer(new char[(size_t)file_size]);
        // copia al buffer
        file.read(buffer.get(), file_size);
        file.close();

        // Procesar cada registro
        for (size_t r = 0; r < num_records; ++r) {
            // obtenemos la etiqueta
            unsigned char lbl = static_cast<unsigned char>(buffer[r * record_bytes]);
            labels.push_back(static_cast<int>(lbl));

            // vector con los pixeles de la imagen, valores entre 0 y 1, basicamente la intensidad del pixel
            vector<float> image(input_dim);
            for (int i = 0; i < input_dim; ++i) {
                // tomamos el pixel
                unsigned char pixel = static_cast<unsigned char>(buffer[r * record_bytes + 1 + i]);
                // normalizamos el valor del pixel a [0-1], ocupa un byte (0-255)
                image[i] = static_cast<float>(pixel) / 255.0f;
            }
            images.push_back(move(image));
        }
    }

};

#endif
