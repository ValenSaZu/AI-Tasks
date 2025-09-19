//  main.cpp
//  AlgoritmosGeneticos
//
//  Created by Amara Barrera, Maria Belen Calle and Camila Salazar on 19/09/25.
//

#include <iostream>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
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
        
        //3. retornar valor de aptitud
        return aptitud = pow(x,2) - pow(y,2) + 2*x*y;
    }
};


//clase poblacion
class poblacion{
public:
    vector<individuo> vPoblacion;
    
    poblacion(){
        for(int i=0; i<20; i++)
            vPoblacion.push_back(individuo()); //inicializar vector
    }
    
    void evaluarAptitudRango(int inicio, int fin){
        for(int i = inicio; i < fin; i++)
            vPoblacion[i].valAptitud();
    }
    
    void evaluarAptitud(){
        const int numThreads = 4;
        const int tamPoblacion = (int)vPoblacion.size();
        const int individuosPorThread = tamPoblacion / numThreads;
                
        vector<thread> threads;
                
        //creando threads
        for(int i = 0; i < numThreads; i++){
            int inicio = i * individuosPorThread;
            int fin;
                    
        //último thread toma cualquier individuo restante
            if(i == numThreads - 1)
                fin = tamPoblacion;
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
    
    //estrategias
    //mutacion
    void mutacion(float prob){
        for(int i=0; i<vPoblacion.size(); i++){
            for(int j=0; j<vPoblacion[i].cromosoma.size(); j++){
                float r = (float) rand()/RAND_MAX;
                if(r < prob){ //intercambiando bits (se puede simplificar si queremos)
                    if(vPoblacion[i].cromosoma[j] == 0)
                        vPoblacion[i].cromosoma[j] = 1;
                    else
                        vPoblacion[i].cromosoma[j] = 0;
                }
            }
        }
    }
    
    //cruzamiento
    /*void cruzamiento(float prob){
        vector<individuo> newGeneration;
        for(int i=0; i<vPoblacion.size(); i+=2){
            individuo p1 = vPoblacion[i];
            individuo p2 = vPoblacion[i+1];
            
            float r = (float)rand()/RAND_MAX;
            
            //haciendo el cruzameinto
            if(r<prob){
                int point = rand() % 13 + 1;
                
                individuo h1, h2;
                h1.cromosoma.clear();
                h2.cromosoma.clear();
                
                //push a la primera mitad
                for(int j=0; j<point; j++){
                    h1.cromosoma.push_back(p1.cromosoma[j]);
                    h2.cromosoma.push_back(p2.cromosoma[j]);
                }
                
                //push a la segunda mitad
                for(int j=point; j<14; j++){
                    h1.cromosoma.push_back(p2.cromosoma[j]);
                    h2.cromosoma.push_back(p1.cromosoma[j]);
                }
                
                //push al vector de nueva generacion
                newGeneration.push_back(h1);
                newGeneration.push_back(h2);
            }
            
            //si no hay cruzamiento
            else{
                newGeneration.push_back(p1);
                newGeneration.push_back(p2);
            }
        }
        
        vPoblacion = newGeneration;
    }
    */
    //completarlo o no, no es necesario
    //void elitismo(){}
    
    void updateGeneration(float probCruzamiento, float probMutacion){
        
        evaluarAptitud();
        
        //cruzamiento(probCruzamiento);
        
        mutacion(probMutacion);
        
        //llama a evaluar Aptitud para calcular las nuevas aptitudes de la nueva generacion
        evaluarAptitud();
    }
};


int main() { //probandooo
    srand((unsigned) time(0));
    poblacion P;

    ofstream file("resultados.csv");
    file << "Generacion,Promedio,Mejor\n";

    for(int gen=0; gen<100; gen++){
        P.updateGeneration(0.7, 0.01);
        float prom = P.promedio();
        float best = P.best();

        cout << "Gen " << gen
             << " | Promedio: " << prom
             << " | Best: " << best << endl;

        file << gen << "," << prom << "," << best << "\n";
    }

    file.close();
    return 0;
}
