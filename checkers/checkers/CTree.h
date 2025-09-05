#ifndef CTree_h
#define CTree_h

#include "CBoard.h"
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

class CTree {
private:
    struct Node {
        CBoard board;
        int value;
        int depth;
        int fromRow, fromCol, toRow, toCol;
        vector<Node*> children;
        
        Node(CBoard b, int d) : board(b), depth(d), value(0) {
            fromRow = fromCol = toRow = toCol = -1;
        }
    };
    
    Node* root;
    int maxDepth;
    
    // Función para evaluar el tablero
    int evaluateBoard(CBoard& board) {
        int score = 0;
        
        // Contar fichas de cada jugador
        int blackPieces = 0;
        int redPieces = 0;
        
        for (int i = 0; i < CBoard::SIZE; i++) {
            for (int j = 0; j < CBoard::SIZE; j++) {
                if (board.board[i][j] == 1) {
                    blackPieces++;
                } else if (board.board[i][j] == 2) {
                    redPieces++;
                }
            }
        }
        
        // Calcular puntuación
        score = blackPieces - redPieces;
        
        return score;
    }
    
    // Función para generar movimientos posibles
    vector<Node*> generateMoves(CBoard& board, int player) {
        vector<Node*> moves;
        
        for (int i = 0; i < CBoard::SIZE; i++) {
            for (int j = 0; j < CBoard::SIZE; j++) {
                if (board.board[i][j] == player) {
                    // Generar movimientos diagonales
                    for (int di = -1; di <= 1; di += 2) {
                        for (int dj = -1; dj <= 1; dj += 2) {
                            int newRow = i + di;
                            int newCol = j + dj;
                            
                            if (newRow >= 0 && newRow < CBoard::SIZE && 
                                newCol >= 0 && newCol < CBoard::SIZE) {
                                
                                if (board.board[newRow][newCol] == 0) {
                                    // Crear nuevo nodo con el movimiento
                                    CBoard newBoard = board;
                                    if (newBoard.movePiece(i, j, newRow, newCol)) {
                                        Node* newNode = new Node(newBoard, 0);
                                        newNode->fromRow = i;
                                        newNode->fromCol = j;
                                        newNode->toRow = newRow;
                                        newNode->toCol = newCol;
                                        moves.push_back(newNode);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return moves;
    }
    
    // Función minimax
    int minimax(Node* node, int depth, bool isMaximizing) {
        if (depth == 0 || node->children.empty()) {
            node->value = evaluateBoard(node->board);
            return node->value;
        }
        
        if (isMaximizing) {
            int maxEval = INT_MIN;
            for (Node* child : node->children) {
                int eval = minimax(child, depth - 1, false);
                maxEval = max(maxEval, eval);
            }
            node->value = maxEval;
            return maxEval;
        } else {
            int minEval = INT_MAX;
            for (Node* child : node->children) {
                int eval = minimax(child, depth - 1, true);
                minEval = min(minEval, eval);
            }
            node->value = minEval;
            return minEval;
        }
    }
    
public:
    CTree() : root(nullptr), maxDepth(3) {}
    
    // Constructor con profundidad personalizada
    CTree(int depth) : root(nullptr), maxDepth(depth) {}
    
    // Destructor
    ~CTree() {
        if (root) {
            delete root;
        }
    }
    
    // Generar árbol de movimientos
    void generateTree(CBoard& board, int player) {
        if (root) {
            delete root;
        }
        
        root = new Node(board, 0);
        generateChildren(root, player, 0);
    }
    
    // Generar hijos de un nodo
    void generateChildren(Node* node, int player, int depth) {
        if (depth >= maxDepth) {
            return;
        }
        
        vector<Node*> moves = generateMoves(node->board, player);
        for (Node* move : moves) {
            node->children.push_back(move);
            generateChildren(move, 1 - player, depth + 1);
        }
    }
    
    // Obtener el mejor movimiento
    Node* getBestMove() {
        if (!root || root->children.empty()) {
            return nullptr;
        }
        
        int bestValue = INT_MIN;
        Node* bestMove = nullptr;
        
        for (Node* child : root->children) {
            int value = minimax(child, maxDepth - 1, false);
            if (value > bestValue) {
                bestValue = value;
                bestMove = child;
            }
        }
        
        return bestMove;
    }
    
    // Obtener el peor movimiento (para el oponente)
    Node* getWorstMove() {
        if (!root || root->children.empty()) {
            return nullptr;
        }
        
        int worstValue = INT_MAX;
        Node* worstMove = nullptr;
        
        for (Node* child : root->children) {
            int value = minimax(child, maxDepth - 1, true);
            if (value < worstValue) {
                worstValue = value;
                worstMove = child;
            }
        }
        
        return worstMove;
    }
    
    // Limpiar el árbol
    void clear() {
        if (root) {
            delete root;
            root = nullptr;
        }
    }
};

#endif