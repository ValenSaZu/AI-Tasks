#ifndef CTree_h
#define CTree_h

#include "CBoard.h"
#include <vector>
#include <climits>
#include <algorithm>

using namespace std;

// Cada nodo representa un estado del tablero después de un movimiento
struct CNode {
    CBoard board;
    vector<CNode*> children;
    int value;
    int startX, startY, endX, endY; // Movimiento que llevó a este estado

    CNode() : value(0), startX(-1), startY(-1), endX(-1), endY(-1) {}
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
        return blackCount - redCount; // Positivo favorece a negras, negativo a rojas
    }

    // Añade un hijo al nodo con un movimiento específico
    void addChild(CNode* parent, int startX, int startY, int endX, int endY, int currentPlayer) {
        // Verifica que la posición sea válida
        if (!parent->board.isValidPosition(endX, endY))
            return;

        // Verificar si la casilla destino está vacía
        if (parent->board.board[endX][endY] != 0)
            return;

        // Crear nuevo tablero clonando el actual
        CBoard newBoard = parent->board;

        // Intentar realizar el movimiento
        if (newBoard.movePiece(startX, startY, endX, endY, currentPlayer)) {
            CNode* child = new CNode();
            child->board = newBoard;
            child->startX = startX;
            child->startY = startY;
            child->endX = endX;
            child->endY = endY;
            parent->children.push_back(child);
        }
    }

    // Crea todos los hijos posibles para un nodo dado
    void createChildren(CNode* node, bool isMaximizingPlayer) {
        for (int i = 0; i < CBoard::SIZE; i++) {
            for (int j = 0; j < CBoard::SIZE; j++) {
                int piece = node->board.board[i][j];

                // Fichas del jugador maximizador (negras)
                if (isMaximizingPlayer && piece == 1) {
                    // Movimientos simples
                    addChild(node, i, j, i + 1, j - 1, 0);
                    addChild(node, i, j, i + 1, j + 1, 0);

                    // Capturas
                    addChild(node, i, j, i + 2, j - 2, 0);
                    addChild(node, i, j, i + 2, j + 2, 0);
                }

                // Fichas del jugador minimizador (rojas)
                if (!isMaximizingPlayer && piece == 2) {
                    // Movimientos simples
                    addChild(node, i, j, i - 1, j - 1, 1);
                    addChild(node, i, j, i - 1, j + 1, 1);

                    // Capturas
                    addChild(node, i, j, i - 2, j - 2, 1);
                    addChild(node, i, j, i - 2, j + 2, 1);
                }
            }
        }
    }

    // Algoritmo MinMax recursivo
    // MAX: La computadora elige el movimiento que le da el mejor resultado
    // MIN: Asume que el jugador elegirá el movimiento que le da el peor resultado
    int minMax(CNode* node, int depth, bool isMaximizingPlayer) {
        // Caso base cuando:
        // - llega al nivel de profundidad del arbol que se ha ingresado para decisiones
        // - ya no hay más movimientos que hacer desde ese punto del tablero
        // - o el juego termina
        // y retorna la heuristica
        if (depth == 0 || node->children.empty() || node->board.isGameOver()) {
            return heuristic(node->board);
        }

        // Para el nivel maximizador
        if (isMaximizingPlayer) {
            // Inicializa con el menor entero posible, así todo numero será mayor de lo que nos dé
            int maxEval = INT_MIN;
            // Evalua cada hijo del nodo
            for (CNode* child : node->children) {
                // Envía para evaluar con false, para que al siguiente evalue min
                int eval = minMax(child, depth - 1, false);
                // Se queda con el mayor valor de los hijos
                maxEval = max(maxEval, eval);
            }
            // El nodo se queda con el mayor valor de los hijos y lo retorna
            node->value = maxEval;
            return maxEval;
        }
        // Igual pero para el minimizador
        else {
            int minEval = INT_MAX;
            for (CNode* child : node->children) {
                int eval = minMax(child, depth - 1, true);
                minEval = min(minEval, eval);
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

    // Encuentra el mejor movimiento usando MinMax
    bool findBestMove(CBoard& currentBoard, int& startX, int& startY, int& endX, int& endY, bool isMaximizingPlayer) {
        root = new CNode();
        root->board = currentBoard;

        createChildren(root, isMaximizingPlayer);

        // Si no hay movimientos posibles
        if (root->children.empty()) {
            clearTree(root);
            root = nullptr;
            return false;
        }

        minMax(root, maxDepth, isMaximizingPlayer);

        // Buscar el mejor movimiento entre los hijos
        int bestValue = isMaximizingPlayer ? INT_MIN : INT_MAX;
        CNode* bestChild = nullptr;

        for (CNode* child : root->children) {
            if ((isMaximizingPlayer && child->value > bestValue) || // Buscamos el movimiento con el valor más ALTO, en caso de ser maximizador
                (!isMaximizingPlayer && child->value < bestValue)) { // Buscamos el movimiento con el valor más BAJO, en caso de ser minimizador
                bestValue = child->value;
                bestChild = child;
            }
        }

        if (bestChild != nullptr) {
            startX = bestChild->startX;
            startY = bestChild->startY;
            endX = bestChild->endX;
            endY = bestChild->endY;
        }

        // Limpiar el árbol después de usarlo
        clearTree(root);
        root = nullptr;

        return bestChild != nullptr;
    }
};

#endif