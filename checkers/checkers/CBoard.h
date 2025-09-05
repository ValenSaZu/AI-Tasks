//
//  CBoard.h
//  ArbolDamas
//
//  Created by Amara Barrera on 4/09/25.
//

#ifndef CBoard_h
#define CBoard_h

#include <cmath>
#include <iostream>

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

    // Verifica si una posición está dentro del tablero
    bool isValidPosition(int x, int y) const {
        return (x >= 0 && x < SIZE && y >= 0 && y < SIZE);
    }

    // Intenta comer una ficha enemiga
    bool capturePiece(int startX, int startY, int endX, int endY, int currentPlayer) {
        int piece = board[startX][startY];
        int dx = endX - startX;
        int dy = endY - startY;

        // Verificar que sea un movimiento de captura (2 casillas en diagonal)
        if (abs(dx) == 2 && abs(dy) == 2) {
            // Obtener la posición de la ficha que sería comida
            int middleX = (startX + endX) / 2;
            int middleY = (startY + endY) / 2;
            int capturedPiece = board[middleX][middleY];

            // Verificar que la ficha capturada sea del oponente y exista
            if (capturedPiece != 0 && capturedPiece != piece) {

                // VERIFICAR DIRECCIÓN DE CAPTURA SEGÚN EL JUGADOR
                // Jugador (negras): solo puede capturar HACIA ARRIBA
                if (currentPlayer == 0 && endX >= startX) {
                    cout << "Error: Fichas negras solo pueden capturar hacia arriba" << endl;
                    return false;
                }
                // Computadora (rojas): solo puede capturar HACIA ABAJO
                if (currentPlayer == 1 && endX <= startX) {
                    cout << "Error: Fichas rojas solo pueden capturar hacia abajo" << endl;
                    return false;
                }

                // Realizar la captura: mover la ficha, eliminar la ficha capturada
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
        if (currentPlayer == 0 && piece != 1) // Jugador humano debe tener ficha negra
            return false;
        if (currentPlayer == 1 && piece != 2) // Computadora debe tener ficha roja
            return false;

        // Comprobar que la casilla destino está vacía
        if (board[endX][endY] != 0)
            return false;

        int dx = endX - startX;
        int dy = endY - startY;

        // PRIMERO intentar captura (las capturas son obligatorias en damas)
        if (abs(dx) == 2 && abs(dy) == 2) {
            return capturePiece(startX, startY, endX, endY, currentPlayer);
        }

        // LUEGO movimiento diagonal simple hacia adelante, sin comer
        if (abs(dx) == 1 && abs(dy) == 1) {
            // Jugador (negras): solo pueden moverse HACIA ARRIBA (disminuyendo fila)
            if (currentPlayer == 0 && endX >= startX) {
                cout << "Error: Fichas negras solo pueden moverse hacia arriba" << endl;
                return false;
            }
            // Computadora (rojas): solo pueden moverse HACIA ABAJO (aumentando fila)
            if (currentPlayer == 1 && endX <= startX) {
                cout << "Error: Fichas rojas solo pueden moverse hacia abajo" << endl;
                return false;
            }

            // Realizar movimiento simple
            board[endX][endY] = piece;
            board[startX][startY] = 0;
            cout << "Movimiento válido: de (" << startX << "," << startY << ") a (" << endX << "," << endY << ")" << endl;
            return true;
        }

        cout << "Error: Movimiento no diagonal válido (dx=" << dx << ", dy=" << dy << ")" << endl;
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

        if (blackCount == 0 && redCount > 0) return 1; // Rojas ganan
        if (redCount == 0 && blackCount > 0) return 0; // Negras ganan
        return -1; // Sin ganador todavía
    }
};

#endif