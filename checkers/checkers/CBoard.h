//
//  CBoard.h
//  ArbolDamas
//
//  Created by Amara Barrera on 4/09/25.
//

#ifndef CBoard_h
#define CBoard_h

#include <cmath>

using namespace std;

class CBoard {
public:
    const static int SIZE = 8;
    int board[SIZE][SIZE];

    // Constructor que inicializa el tablero
    CBoard() {
        initializeBoard();
    }

    // Inicializar el tablero con la configuración inicial
    void initializeBoard() {
        // Limpiar el tablero
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                board[i][j] = 0;
            }
        }
        
        // Colocar fichas negras (jugador) en las filas 5, 6, 7
        for (int i = 5; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if ((i + j) % 2 == 1) {
                    board[i][j] = 1; // Fichas negras
                }
            }
        }
        
        // Colocar fichas rojas (computadora) en las filas 0, 1, 2
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < SIZE; j++) {
                if ((i + j) % 2 == 1) {
                    board[i][j] = 2; // Fichas rojas
                }
            }
        }
    }

    // Verificar si un movimiento es válido
    bool isValidMove(int fromRow, int fromCol, int toRow, int toCol) {
        // Verificar que las coordenadas estén dentro del tablero
        if (fromRow < 0 || fromRow >= SIZE || fromCol < 0 || fromCol >= SIZE ||
            toRow < 0 || toRow >= SIZE || toCol < 0 || toCol >= SIZE) {
            return false;
        }
        
        // Verificar que la casilla de origen tenga una ficha
        if (board[fromRow][fromCol] == 0) {
            return false;
        }
        
        // Verificar que la casilla de destino esté vacía
        if (board[toRow][toCol] != 0) {
            return false;
        }
        
        // Verificar que sea un movimiento diagonal
        int rowDiff = abs(toRow - fromRow);
        int colDiff = abs(toCol - fromCol);
        
        if (rowDiff != colDiff) {
            return false;
        }
        
        // Verificar que sea un movimiento de una casilla (por ahora)
        if (rowDiff != 1) {
            return false;
        }
        
        // Verificar dirección del movimiento según el jugador
        if (board[fromRow][fromCol] == 1) { // Ficha negra
            if (toRow >= fromRow) { // Las negras deben moverse hacia arriba
                return false;
            }
        } else if (board[fromRow][fromCol] == 2) { // Ficha roja
            if (toRow <= fromRow) { // Las rojas deben moverse hacia abajo
                return false;
            }
        }
        
        return true;
    }

    // Mover una ficha
    bool movePiece(int fromRow, int fromCol, int toRow, int toCol) {
        if (isValidMove(fromRow, fromCol, toRow, toCol)) {
            board[toRow][toCol] = board[fromRow][fromCol];
            board[fromRow][fromCol] = 0;
            return true;
        }
        return false;
    }

    // Verificar si el juego ha terminado
    bool isGameOver() {
        // Verificar si quedan fichas de algún jugador
        bool hasBlack = false;
        bool hasRed = false;
        
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (board[i][j] == 1) hasBlack = true;
                if (board[i][j] == 2) hasRed = true;
            }
        }
        
        return !hasBlack || !hasRed;
    }

    // Obtener el ganador
    int getWinner() {
        if (isGameOver()) {
            bool hasBlack = false;
            bool hasRed = false;
            
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    if (board[i][j] == 1) hasBlack = true;
                    if (board[i][j] == 2) hasRed = true;
                }
            }
            
            if (hasBlack && !hasRed) return 1; // Jugador gana
            if (!hasBlack && hasRed) return 2; // Computadora gana
        }
        return 0; // Juego en progreso
    }
};

#endif
