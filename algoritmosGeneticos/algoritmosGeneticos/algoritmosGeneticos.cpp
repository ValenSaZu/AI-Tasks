//  algoritmosGeneticos_mejorado.cpp
//  AlgoritmosGeneticos
//
//  Created by Amara Barrera, Maria Belen Calle and Camila Salazar on 19/09/25.
//  Modificado para mejores resultados de convergencia
//

#include <iostream>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>
using namespace std;

class individuo{
public:
    vector<int> cromosoma; //individuos con cromosomas de 14
    float x, y;
    float aptitud;
    
    individuo(){
        for (int i=0; i < 14; i++)
            cromosoma.push_back(rand() % 2);
        
        x = 0.0;
        y = 0.0;
        aptitud = 0.0;
    }
    
    float valAptitud(){
        //1. separar cadena de bits para X y Y
        //2. decodificar cadenas de bits
        int valorX = 0;
        int valorY = 0;
        int exponent = 6;
        
        for(int i=0; i<7; i++){
            valorX += cromosoma[i] * pow(2,exponent);
            exponent--;
        }
        
        exponent = 6;
        for(int i = 7; i<14; i++){
            valorY += cromosoma[i] * pow(2, exponent);
            exponent--;
        }
        
        x = -63 + (valorX/127.0) * 126;
        y = -63 + (valorY/127.0) * 126;
        
        //3. retornar valor de aptitud para minimización
        return aptitud = pow(x,2) - pow(y,2) + 2*x*y;
    }
};

//clase poblacion
class poblacion{
public:
    vector<individuo> vPoblacion;
    int tamanoPoblacion;
    
    poblacion(int tamano = 50){
        tamanoPoblacion = tamano;
        for(int i=0; i<tamanoPoblacion; i++)
            vPoblacion.push_back(individuo()); //inicializar vector
    }
    
    void evaluarAptitudRango(int inicio, int fin){
        for(int i = inicio; i < fin; i++)
            vPoblacion[i].valAptitud();
    }
    
    void evaluarAptitud(){
        const int numThreads = 4;
        const int individuosPorThread = tamanoPoblacion / numThreads;
                
        vector<thread> threads;
                
        //creando threads
        for(int i = 0; i < numThreads; i++){
            int inicio = i * individuosPorThread;
            int fin;
                    
            //último thread toma cualquier individuo restante
            if(i == numThreads - 1)
                fin = tamanoPoblacion;
            else
                fin = (i + 1) * individuosPorThread;
                    
            threads.emplace_back(&poblacion::evaluarAptitudRango, this, inicio, fin);
        }
                
        for(auto& t : threads){
            t.join();
        }
    }
    
    float promedio(){
        float suma = 0;
        for(int i=0; i<vPoblacion.size(); i++){
            suma += vPoblacion[i].aptitud;
        }
        return suma/vPoblacion.size();
    }
    
    float best(){
        float theBest = vPoblacion[0].aptitud;
        for(int i=0; i<vPoblacion.size(); i++){
            if(vPoblacion[i].aptitud < theBest)
                theBest = vPoblacion[i].aptitud;
        }
        return theBest;
    }
    
    // Selección por torneo para mejorar convergencia
    individuo seleccionTorneo(){
        int tamanoTorneo = 3;
        individuo mejor = vPoblacion[rand() % tamanoPoblacion];
        
        for(int i = 1; i < tamanoTorneo; i++){
            individuo competidor = vPoblacion[rand() % tamanoPoblacion];
            if(competidor.aptitud < mejor.aptitud){
                mejor = competidor;
            }
        }
        return mejor;
    }
    
    //mutación mejorada
    void mutacion(float prob){
        for(int i=0; i<vPoblacion.size(); i++){
            for(int j=0; j<vPoblacion[i].cromosoma.size(); j++){
                float r = (float) rand()/RAND_MAX;
                if(r < prob){
                    vPoblacion[i].cromosoma[j] = 1 - vPoblacion[i].cromosoma[j];
                }
            }
        }
    }
    
    //cruzamiento mejorado con selección por torneo
    void cruzamiento(float prob){
        vector<individuo> newGeneration;
        
        // Elitismo: preservar al mejor individuo
        individuo mejor = vPoblacion[0];
        for(int i = 1; i < tamanoPoblacion; i++){
            if(vPoblacion[i].aptitud < mejor.aptitud){
                mejor = vPoblacion[i];
            }
        }
        newGeneration.push_back(mejor);
        
        // Generar el resto de la población
        while(newGeneration.size() < tamanoPoblacion){
            individuo p1 = seleccionTorneo();
            individuo p2 = seleccionTorneo();
            
            float r = (float)rand()/RAND_MAX;
            
            if(r < prob && newGeneration.size() + 1 < tamanoPoblacion){
                // Cruzamiento de dos puntos
                int point1 = rand() % 13 + 1;
                int point2 = rand() % 13 + 1;
                if(point1 > point2) swap(point1, point2);
                
                individuo h1, h2;
                h1.cromosoma.clear();
                h2.cromosoma.clear();
                
                // Primera parte
                for(int j=0; j<point1; j++){
                    h1.cromosoma.push_back(p1.cromosoma[j]);
                    h2.cromosoma.push_back(p2.cromosoma[j]);
                }
                
                // Parte media (intercambiada)
                for(int j=point1; j<point2; j++){
                    h1.cromosoma.push_back(p2.cromosoma[j]);
                    h2.cromosoma.push_back(p1.cromosoma[j]);
                }
                
                // Última parte
                for(int j=point2; j<14; j++){
                    h1.cromosoma.push_back(p1.cromosoma[j]);
                    h2.cromosoma.push_back(p2.cromosoma[j]);
                }
                
                newGeneration.push_back(h1);
                if(newGeneration.size() < tamanoPoblacion){
                    newGeneration.push_back(h2);
                }
            } else {
                newGeneration.push_back(p1);
                if(newGeneration.size() < tamanoPoblacion){
                    newGeneration.push_back(p2);
                }
            }
        }
        
        // Asegurar que la nueva generación tenga el tamaño correcto
        while(newGeneration.size() > tamanoPoblacion){
            newGeneration.pop_back();
        }
        
        vPoblacion = newGeneration;
    }
    
    void updateGeneration(float probCruzamiento, float probMutacion){
        evaluarAptitud();
        cruzamiento(probCruzamiento);
        mutacion(probMutacion);
        evaluarAptitud();
    }
};

int main() {
    // Semilla fija para resultados reproducibles
    srand(42);
    
    // Población más grande para mejor convergencia
    poblacion P(50);

    ofstream file("resultados.csv");
    file << "Generacion,Promedio,Mejor\n";

    cout << "Iniciando Algoritmo Genético para minimizar f(x,y) = x² - y² + 2xy" << endl;
    cout << "Población: 50 individuos, Generaciones: 100" << endl;
    cout << "Rango: x,y ∈ [-63, 63]" << endl << endl;

    for(int gen=0; gen<100; gen++){
        // Parámetros ajustados para mejor convergencia
        P.updateGeneration(0.8, 0.02);
        float prom = P.promedio();
        float best = P.best();

        cout << "Gen " << gen
             << " | Promedio: " << prom
             << " | Mejor: " << best << endl;

        file << gen << "," << prom << "," << best << "\n";
    }

    file.close();
    
    cout << "\nOptimización completada. Archivo 'resultados.csv' generado." << endl;
    cout << "Ejecuta 'python graficar.py' para ver la gráfica." << endl;
    
    return 0;
}