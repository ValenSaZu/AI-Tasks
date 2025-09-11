//
//  coloreo.h
//  ColoreoGrafos
//
//  Created by Amara Barrera on 11/09/25.
//

#include <iostream>
#include <vector>
using namespace std;

class Coloreo{
public:
    int numVertices;
    vector<vector<int>> nodos;
    vector<int> Colores;
    
    Coloreo(int numVert): numVertices(numVert), Colores(numVert, -1){
        nodos.resize(numVert);
    }
    
    bool esValido(int nodo, int color) {
        for(int i = 0; i < nodos[nodo].size(); i++) {
            int vecino = nodos[nodo][i];
            if(Colores[vecino] == color) {
                return false;
            }
        }
        return true;
    }
    
    //sin heuristicas
    void BT(int current, int numColors){
        bool esValido;
        if(current == numVertices)
            return;
       
        for(int color = 0; color < numColors; color++){
            if(esValido(current, color)){
                Colores[current] = color;
                        
                if(BT(current + 1, numColors))
                    return true;
                Colores[current] = -1; // Backtrack
            }
        }
        return false;
    }
    
    //estos llevan la mas restringida y restrictiva
    void BTFC(){}
    void MAC(){}
};
