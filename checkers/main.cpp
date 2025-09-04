//
//  main.cpp
//  ArbolDamas
//
//  Created by Amara Barrera on 4/09/25.
//

#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

//en el tablero aparecen las opciones de mover, comer, contar fichas
struct CTablero{
    static const int size = 8;
    int board[size][size];
    
    CTablero(){ //constructor inicializando tablero vacio
        for(int i = 0; i<size; i++){
            for(int j = 0; j<size; j++){
                board[i][j] = 0;
            }
        }
        
        //inicializar fichas negras
        for(int i = 0; i<3; i++){
            for(int j = 0; j<size; j++){
                if((i+j) % 2 == 1)
                    board[i][j] = 1;
            }
        }
        
        //inicializar fichas rojas
        for(int i = size-3; i<size; i++){
            for(int j = 0; j<size; j++){
                if((i+j) % 2 == 1)
                    board[i][j] = 2;
            }
        }
    }
    
    //esto se puede poner dentro de mover fichas si quieren
    int comer(int Xi, int Yi, int Xf, int Yf, int turno){
        int ficha = board[Xi][Yi];
        int dx = Xf-Xi;
        int dy = Yf-Yi;
        
        if(abs(dx) == 2 && abs(dy) == 2){
            int xm = (Xi + Xf)/2;
            int ym = (Yi + Yf)/2;
            int fichaCapturada = board[xm][ym];
            
            if(fichaCapturada != 0 && fichaCapturada != ficha){
                board[Xf][Yf] = ficha;
                board[Xi][Yi] = 0;
                board[xm][ym] = 0;
                return 1;
            }
        }
        return 0;
    };
    
    int moverFichas(int Xi, int Yi, int Xf, int Yf, int turno){
        if(!(Xi >= 0 && Xi < 8 && Yi >= 0 && Yi < 8 && Xf >=0 && Xf < 8 &&  Yf >= 0 && Yf < 8))
            return 0;
        
        int ficha = board[Xi][Yi];
        if(ficha == 0) //vacio
            return 0;
        
        //comprueba fichas correctas del jugador
        if(turno == 0 && ficha != 1)
            return 0;
        if(turno == 1 && ficha != 2)
            return 0;
        
        //comprobando que el lugar al que se mueve esta vacio
        if(board[Xf][Yf] != 0)
            return 0;
        
        int dx = Xf-Xi;
        int dy = Yf-Yi;
            
        //movimiento diagonal adelante
        if(abs(dx) == 1 && abs(dy) == 1){
            board[Xf][Yf] = ficha;
            board[Xi][Yi] = 0;
            turno = 1 - turno; //cambia el turno del jugador
            return 1;
        }
            
        //movimiento para comer
        if(comer(Xi, Yi, Xf, Yf, turno)){
            turno = 1 - turno; //cambia el turno del jugador
            return 1;
        }

        return 0;
    }
    
    void contarFichas(int &numNegras, int &numRojas){ //0 si vacio, 1 si es negra, 2 si es roja
        numNegras = 0;
        numRojas = 0;
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                //1 si hay una negra
                if(board[i][j] == 1)
                    numNegras++;
                //2 si hay una roja
                else if (board[i][j] == 2)
                    numRojas++;
            }
        }
    };
};

//cada nodo representa un tablero despues de un movimiento
struct CNodo{
    CTablero tablero;
    vector<CNodo*> hijos;
    int valor;
};

//en CArbol se implementa el algoritmo MinMax
struct CArbol{
    CNodo* root;
    int nivel;
    
    CArbol(){
        root = nullptr;
    }
    
    int heuristica(int numNegras, int numRojas){
        return numNegras - numRojas;
    }
    
    void addHijo(CNodo* nodo, int Xi, int Yi, int Xf, int Yf){
        if (Xf < 0 || Xf >= 8 || Yf < 0 || Yf >= 8)
            return;
        
        //verificar si direccion destino esta vacia
        if (nodo->tablero.board[Xf][Yf] != 0)
            return;
        
        //clonando tablero
        CTablero newTablero = nodo->tablero;
        newTablero.board[Xf][Yf] = newTablero.board[Xi][Yi];
        newTablero.board[Xi][Yi] = 0;
        
        //creando nodo hijo
        CNodo* hijo = new CNodo();
        hijo->tablero = newTablero;
        nodo->hijos.push_back(hijo);
    }
    
    void creaHijos(CNodo* nodo, bool esMax){
        for (int i = 0; i<8; i++){
            for (int j = 0; j<8; j++){
                int ficha = nodo->tablero.board[i][j];
                
                if(esMax && ficha == 1){
                    addHijo(nodo, i, j, i+1, j-1);
                    addHijo(nodo, i, j, i+1, j+1);
                }
                
                if(!esMax && ficha == 2){
                    addHijo(nodo, i, j, i-1, j-1);
                    addHijo(nodo, i, j, i-1, j+1);
                }
            }
        }
    }
    
    int MinMax(CNodo* nodoCurrent, int depth, bool esMax){//es recursivo
        //si se llega a la profundidad max -> evalu con heuristica
        //si el juego termina -> evaluar estado final
        if (esMax){
            MinMax(nodoCurrent, depth-1, false);
        }
        
        return 1;
    };
};

int main() {
    return 0;
}
