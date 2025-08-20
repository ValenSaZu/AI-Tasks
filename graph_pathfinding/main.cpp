//
//  main.cpp
//  BusquedasCiegas
//
//  Created by Amara Barrera on 20/08/25.
//

//Grafo como vector de vectores. Cada indice del vector principal contiene un vector que guarda los vertices a los que esta conectado el vertice del indice. Ejemplo: [0,1,2,3] -> por dentro se ve como [[1,2], [0,3], [0,3], [1,2]] => entonces el indice 0 que tiene el vertice 0 esta conectado a los vertices 1 y 2; el indice 1 que contiene al vertice 1, esta conectado a los vertices 0 y 3. Asi sucesivamente

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
using namespace std;

//Clase Grafo
template <class T>
class CGraph{
private:
    int numVertices = 0;
    
public:
    vector<vector<T>> vertices;
    CGraph() = default; //para dinamicos
    
    CGraph(int size){ //para grafos fijos
        Resize(size);
    }
    
    //agrega vertices
    void AddVertex(){
        vertices.push_back(vector<T>());
        numVertices++;
    };
    
    //crea arista entre i y j
    void Connections(T i, T j){
        if(i<0 || i>=numVertices || j<0 || j>=numVertices){
            cout << "fuera de rango" << endl;
            return;
        }
        
        if(i==j){
            cout << "fuera de rango" << endl;
            return;
        }
        
        vertices[i].push_back(j);
        vertices[j].push_back(i);
    };
    
    //para ponerle size especifico al grafo
    void Resize(int n){
        if(n<0)
            cout << "no valido" << endl;
        vertices.resize(n);
        numVertices = n;
    }
    
    //algoritmo de busqueda por amplitud
    //recorre los vecinos directos y luego los vecinos de esos vecinos. Agrega vecinos al final de la cola.
    void BFS(int start = 0){
        vector<bool> visitados(numVertices, false);
        vector<int> anteriores (numVertices, -1);
        queue<int> cola;
        
        visitados[start] = true;
        cola.push(start);
        
        while(!cola.empty()){
            int vertActual = cola.front();
            cola.pop();
            cout << vertActual << " ";
            
            for(int i = 0; i<vertices[vertActual].size(); i++){
                int vecino = vertices[vertActual][i];
                if(!visitados[vecino]){
                    visitados[vecino] = true;
                    anteriores[vecino] = vertActual;
                    cola.push(vecino); //agregando vecinos a la cola
                }
            }
        }
    }
    
    //algoritmo de busqueda por profundidad
    //recorre el grafo yendo a lo mas profundo y cuando termina retrocede. Agrega al inicio
    void DFS(int start = 0){
        vector<bool> visitados(numVertices, false);
        vector<int> anteriores(numVertices, -1);
        stack<int> pila;
        
        pila.push(start);
        
        while(!pila.empty()){
            int vertAcual = pila.top();
            pila.pop();
            
            if(!visitados[vertAcual]){
                visitados[vertAcual] = true;
                cout << vertAcual << " ";
            }
            
            for(int i = vertices[vertAcual].size()-1; i >= 0; i--){
                int vecino = vertices[vertAcual][i];
                
                if(!visitados[vecino]){
                    anteriores[vecino] = vertAcual;
                    pila.push(vecino);
                }
            }
        }
    }
    
};
  
//funcion para crear las conexiones en tablero (si quieres no la usas Vale)
void createConnections(CGraph<int>& grafo, int width, int height) {
        
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int actual = y * width + x; //ubica al vertice
                
            //conecta con vecino derecho
            if(x + 1 < width) {
                int right = actual + 1;
                grafo.Connections(actual, right);
            }
                
            //conecta con vecino abajo
            if(y + 1 < height) {
                int down = actual + width;
                grafo.Connections(actual, down);
            }
            
            //conecta con diagonal derecha
            if(x+1 < width && y+1 < height){
                int diagonal = actual + width + 1;
                grafo.Connections(actual, diagonal);
            }
            
            //conecta con diagonal izq
            if(x-1 >= 0 && y+1 < height){
                int diagonal = actual + width - 1;
                grafo.Connections(actual, diagonal);
            }
        }
    }
}

//convertir coordenadas a indices
int coordToIndex(int x, int y, int width) {
    return y * width + x;
}

int main() {
    CGraph<int> grafo(10000);
    createConnections(grafo, 100, 100);
    return 0;
}
