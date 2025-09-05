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

    // Constructor que inicializa el tablero vacío
    CBoard() {
        initializeBoard();
    }

    void initializeBoard() {
        // Primero inicializar todo a 0 (vacío)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                board[i][j] = 0;
            }
        }
        
        // Fichas NEGRAS (jugador) - primeras 3 filas (0, 1, 2)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < SIZE; j++) {
                if ((i + j) % 2 == 1) {  // Solo en casillas oscuras
                    board[i][j] = 1; // Fichas negras (jugador)
                }
            }
        }
        
        // Fichas ROJAS (computadora) - últimas 3 filas (5, 6, 7)
        for (int i = 5; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if ((i + j) % 2 == 1) {  // Solo en casillas oscuras
                    board[i][j] = 2; // Fichas rojas (computadora)
                }
            }
        }
}

    // Verifica si una posición está dentro del tablero
    bool isValidPosition(int x, int y) const {
        return (x >= 0 && x < SIZE && y >= 0 && y < SIZE);
    }

    // Intenta comer una ficha enemiga
    bool capturePiece(int startX, int startY, int endX, int endY, int currentPlayer) {
        int piece = board[startX][startY];
        int dx = endX - startX;
        int dy = endY - startY;

        if (abs(dx) == 2 && abs(dy) == 2) {
            // Obtenemos la posicion de la ficha que en este caso sería comida
            // (está en medio de donde inicia y donde terminará luego del movimiento)
            int middleX = (startX + endX) / 2;
            int middleY = (startY + endY) / 2;
            int capturedPiece = board[middleX][middleY];

            // Verificar que la ficha capturada sea del oponente
            // Coloca la ficha en el lugar del movimiento, libera el lugar donde estaba la ficha que come y la ficha comida
            if (capturedPiece != 0 && capturedPiece != piece) {
                board[endX][endY] = piece;
                board[startX][startY] = 0;
                board[middleX][middleY] = 0;
                return true;
            }
        }
        return false;
    }

    // Mueve una ficha en el tablero
    bool movePiece(int startX, int startY, int endX, int endY, int currentPlayer) {
        // Verificar posiciones válidas
        if (!isValidPosition(startX, startY) || !isValidPosition(endX, endY))
            return false;

        int piece = board[startX][startY];
        if (piece == 0) // Casilla vacía
            return false;

        // Comprobar que la ficha pertenece al jugador actual
        if (currentPlayer == 0 && piece != 1)
            return false;
        if (currentPlayer == 1 && piece != 2)
            return false;

        // Comprobar que la casilla destino está vacía
        if (board[endX][endY] != 0)
            return false;

        int dx = endX - startX;
        int dy = endY - startY;

        // Movimiento diagonal simple hacia adelante, o sea sin comer
        if (abs(dx) == 1 && abs(dy) == 1) {
            // Para fichas normales, verificar dirección correcta
            if (currentPlayer == 0 && dx < 0) return false; // Negras solo avanzan hacia abajo
            if (currentPlayer == 1 && dx > 0) return false; // Rojas solo avanzan hacia arriba
            // Coloca la pieza done va y libera el lugar donde estaba
            board[endX][endY] = piece;
            board[startX][startY] = 0;
            return true;
        }

        // Movimiento para comer
        if (capturePiece(startX, startY, endX, endY, currentPlayer)) {
            return true;
        }

        return false;
    }

    // Cuenta las fichas de cada jugador por todo el tablero
    void countPieces(int& blackCount, int& redCount) const {
        blackCount = 0;
        redCount = 0;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (board[i][j] == 1)
                    blackCount++;
                else if (board[i][j] == 2)
                    redCount++;
            }
        }
    }

    // Verifica si el juego ha terminado, termina cuando alguien se queda sin fichas
    bool isGameOver() const {
        int blackCount, redCount;
        countPieces(blackCount, redCount);
        return blackCount == 0 || redCount == 0;
    }

    // Obtiene el ganador (0: negras, 1: rojas, -1: empate o sin ganador)
    int getWinner() const {
        int blackCount, redCount;
        countPieces(blackCount, redCount);

        if (blackCount == 0 && redCount > 0) return 1;
        if (redCount == 0 && blackCount > 0) return 0;
        return -1;
    }
};

#endif
