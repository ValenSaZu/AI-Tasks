#ifndef CTree_h
#define CTree_h

#include "CBoard.h"
#include <vector>
#include <climits>
#include <algorithm>
#include <iostream>

using namespace std;

// Cada nodo representa un estado del tablero después de un movimiento
struct CNode {
    CBoard board;
    vector<CNode*> children;
    int value;
    int fromRow, fromCol, toRow, toCol; // Movimiento que llevó a este estado

    CNode() : value(0), fromRow(-1), fromCol(-1), toRow(-1), toCol(-1) {}
};

// Implementa el algoritmo MinMax para el árbol de decisiones
class CTree {
private:
    CNode* root;
    int maxDepth;

    // Limpia el árbol recursivamente
    void clearTree(CNode* node) {
        if (node == nullptr) return;

        for (CNode* child : node->children) {
            clearTree(child);
        }
        delete node;
    }

    // Calcula la heurística para un estado del tablero, se basa en la cantidad de fichas
    int heuristic(const CBoard& board) const {
        int blackCount, redCount;
        board.countPieces(blackCount, redCount);
        // Positivo favorece a rojas (computadora), negativo favorece a negras (jugador)
        return redCount - blackCount;
    }

    // Genera todos los movimientos posibles para una ficha específica
    void generateMovesForPiece(CNode* node, int row, int col, int currentPlayer) {
        // Direcciones para movimientos normales y capturas
        int directions[4][2] = { {1, -1}, {1, 1}, {-1, -1}, {-1, 1} };
        int captureDirections[4][2] = { {2, -2}, {2, 2}, {-2, -2}, {-2, 2} };

        // PRIMERO verificar capturas (son obligatorias en damas)
        for (int i = 0; i < 4; i++) {
            int newRow = row + captureDirections[i][0];
            int newCol = col + captureDirections[i][1];
            int middleRow = row + captureDirections[i][0] / 2;
            int middleCol = col + captureDirections[i][1] / 2;

            // Verificar que la posición final y la intermedia sean válidas
            if (node->board.isValidPosition(newRow, newCol) &&
                node->board.board[newRow][newCol] == 0 && // Casilla destino vacía
                node->board.isValidPosition(middleRow, middleCol)) {

                int middlePiece = node->board.board[middleRow][middleCol];
                // Verificar que la ficha del medio sea del oponente
                if (middlePiece != 0 && middlePiece != node->board.board[row][col]) {

                    // VERIFICAR DIRECCIÓN DE CAPTURA VÁLIDA
                    bool validCapture = false;
                    // Jugador (negras): solo puede capturar HACIA ARRIBA
                    if (currentPlayer == 0 && newRow < row) {
                        validCapture = true;
                    }
                    // Computadora (rojas): solo puede capturar HACIA ABAJO
                    if (currentPlayer == 1 && newRow > row) {
                        validCapture = true;
                    }

                    if (validCapture) {
                        CBoard newBoard = node->board;
                        if (newBoard.movePiece(row, col, newRow, newCol, currentPlayer)) {
                            CNode* child = new CNode();
                            child->board = newBoard;
                            child->fromRow = row;
                            child->fromCol = col;
                            child->toRow = newRow;
                            child->toCol = newCol;
                            node->children.push_back(child);
                        }
                    }
                }
            }
        }

        // SOLO si no hay capturas, verificar movimientos normales
        if (node->children.empty()) {
            for (int i = 0; i < 4; i++) {
                int newRow = row + directions[i][0];
                int newCol = col + directions[i][1];

                if (node->board.isValidPosition(newRow, newCol) &&
                    node->board.board[newRow][newCol] == 0) { // Casilla vacía

                    // Jugador (negras): solo pueden moverse HACIA ARRIBA (disminuyendo fila)
                    if (currentPlayer == 0 && directions[i][0] > 0) continue;
                    // Computadora (rojas): solo pueden moverse HACIA ABAJO (aumentando fila)
                    if (currentPlayer == 1 && directions[i][0] < 0) continue;

                    CBoard newBoard = node->board;
                    if (newBoard.movePiece(row, col, newRow, newCol, currentPlayer)) {
                        CNode* child = new CNode();
                        child->board = newBoard;
                        child->fromRow = row;
                        child->fromCol = col;
                        child->toRow = newRow;
                        child->toCol = newCol;
                        node->children.push_back(child);
                    }
                }
            }
        }
    }

    // Crea todos los hijos posibles para un nodo dado
    void createChildren(CNode* node, int currentPlayer) {
        // Recorrer todo el tablero buscando fichas del jugador actual
        for (int i = 0; i < CBoard::SIZE; i++) {
            for (int j = 0; j < CBoard::SIZE; j++) {
                if (node->board.board[i][j] == (currentPlayer == 0 ? 1 : 2)) {
                    generateMovesForPiece(node, i, j, currentPlayer);
                }
            }
        }
    }

    // Algoritmo MinMax recursivo con poda alpha-beta
    // MAX: La computadora elige el movimiento que le da el mejor resultado
    // MIN: Asume que el jugador elegirá el movimiento que le da el peor resultado
    int minMax(CNode* node, int depth, bool isMaximizingPlayer, int alpha, int beta) {
        // Caso base cuando:
        // - llega al nivel de profundidad del árbol que se ha ingresado para decisiones
        // - ya no hay más movimientos que hacer desde ese punto del tablero
        // - o el juego termina
        // y retorna la heurística
        if (depth == 0 || node->children.empty() || node->board.isGameOver()) {
            return heuristic(node->board);
        }

        // Para el nivel maximizador (computadora - rojas)
        if (isMaximizingPlayer) {
            int maxEval = INT_MIN;
            // Evalúa cada hijo del nodo
            for (CNode* child : node->children) {
                // Envía para evaluar con false, para que al siguiente evalúe min
                int eval = minMax(child, depth - 1, false, alpha, beta);
                // Se queda con el mayor valor de los hijos
                maxEval = max(maxEval, eval);
                alpha = max(alpha, eval);
                // Poda alpha-beta: si beta es menor o igual a alpha, cortar la búsqueda
                if (beta <= alpha) break;
            }
            // El nodo se queda con el mayor valor de los hijos y lo retorna
            node->value = maxEval;
            return maxEval;
        }
        // Para el nivel minimizador (jugador - negras)
        else {
            int minEval = INT_MAX;
            for (CNode* child : node->children) {
                int eval = minMax(child, depth - 1, true, alpha, beta);
                minEval = min(minEval, eval);
                beta = min(beta, eval);
                // Poda alpha-beta: si beta es menor o igual a alpha, cortar la búsqueda
                if (beta <= alpha) break;
            }
            node->value = minEval;
            return minEval;
        }
    }

public:
    CTree() : root(nullptr), maxDepth(3) {}

    ~CTree() {
        clearTree(root);
    }

    // Establece la profundidad máxima del árbol
    void setMaxDepth(int depth) {
        maxDepth = depth;
    }

    // Genera el árbol de decisiones a partir del tablero actual
    void generateTree(const CBoard& currentBoard, int currentPlayer) {
        // Limpiar árbol anterior si existe
        clearTree(root);

        // Crear nuevo nodo raíz
        root = new CNode();
        root->board = currentBoard;

        // Generar todos los movimientos posibles
        createChildren(root, currentPlayer);

        // Si hay movimientos posibles, ejecutar MinMax
        if (!root->children.empty()) {
            minMax(root, maxDepth, currentPlayer == 1, INT_MIN, INT_MAX);
        }
    }

    // Obtiene el mejor movimiento según el algoritmo MinMax
    CNode* getBestMove() {
        if (root == nullptr || root->children.empty()) {
            return nullptr;
        }

        // Buscar el hijo con el mejor valor
        CNode* bestChild = root->children[0];
        for (CNode* child : root->children) {
            if (child->value > bestChild->value) {
                bestChild = child;
            }
        }

        return bestChild;
    }
};

#endif