#ifndef CGraphics_h
#define CGraphics_h

#include <GL/glut.h>
#include "CBoard.h"
#include "CTree.h"
#include <iostream>
#include <string>
#include <cmath>

using namespace std;

// Variables globales
CBoard gameBoard;
CTree gameTree;
// 0: jugador humano (NEGRAS)
// 1: computadora (ROJAS)
int currentPlayer = 0;
int selectedX = -1, selectedY = -1;
bool isSelected = false;
int difficulty = 3;

// Función para convertir coordenadas del mouse a coordenadas del tablero
void mouseToBoard(int x, int y, int& row, int& col) {
    // Ajustar las coordenadas para que el tablero esté centrado
    int boardSize = 400;
    int startX = (glutGet(GLUT_WINDOW_WIDTH) - boardSize) / 2;
    int startY = (glutGet(GLUT_WINDOW_HEIGHT) - boardSize) / 2;
    
    // Convertir coordenadas del mouse a coordenadas del tablero
    col = (x - startX) / (boardSize / 8);
    row = (y - startY) / (boardSize / 8);
    
    // Asegurar que estén dentro del rango válido
    if (col < 0) col = 0;
    if (col >= 8) col = 7;
    if (row < 0) row = 0;
    if (row >= 8) row = 7;
}

// Función para dibujar el tablero
void drawBoard() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    int boardSize = 400;
    int startX = (glutGet(GLUT_WINDOW_WIDTH) - boardSize) / 2;
    int startY = (glutGet(GLUT_WINDOW_HEIGHT) - boardSize) / 2;
    int cellSize = boardSize / 8;
    
    // Dibujar el tablero
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // Alternar colores de las casillas
            if ((i + j) % 2 == 0) {
                glColor3f(0.8f, 0.6f, 0.4f); // Color claro
            } else {
                glColor3f(0.4f, 0.2f, 0.0f); // Color oscuro
            }
            
            // Dibujar casilla
            glBegin(GL_QUADS);
            glVertex2f(startX + j * cellSize, startY + i * cellSize);
            glVertex2f(startX + (j + 1) * cellSize, startY + i * cellSize);
            glVertex2f(startX + (j + 1) * cellSize, startY + (i + 1) * cellSize);
            glVertex2f(startX + j * cellSize, startY + (i + 1) * cellSize);
            glEnd();
            
            // Dibujar ficha si existe
            if (gameBoard.board[i][j] != 0) {
                int centerX = startX + j * cellSize + cellSize / 2;
                int centerY = startY + i * cellSize + cellSize / 2;
                int radius = cellSize / 3;
                
                // Color de la ficha
                if (gameBoard.board[i][j] == 1) {
                    glColor3f(0.0f, 0.0f, 0.0f); // Negro
                } else if (gameBoard.board[i][j] == 2) {
                    glColor3f(0.8f, 0.0f, 0.0f); // Rojo
                }
                
                // Dibujar círculo
                glBegin(GL_TRIANGLE_FAN);
                glVertex2f(centerX, centerY);
                for (int k = 0; k <= 360; k += 10) {
                    float angle = k * 3.14159f / 180.0f;
                    glVertex2f(centerX + radius * cos(angle), centerY + radius * sin(angle));
                }
                glEnd();
            }
            
            // Dibujar selección si existe
            if (isSelected && selectedX == j && selectedY == i) {
                glColor3f(0.0f, 1.0f, 0.0f); // Verde
                glLineWidth(3.0f);
                glBegin(GL_LINE_LOOP);
                glVertex2f(startX + j * cellSize + 2, startY + i * cellSize + 2);
                glVertex2f(startX + (j + 1) * cellSize - 2, startY + i * cellSize + 2);
                glVertex2f(startX + (j + 1) * cellSize - 2, startY + (i + 1) * cellSize - 2);
                glVertex2f(startX + j * cellSize + 2, startY + (i + 1) * cellSize - 2);
                glEnd();
                glLineWidth(1.0f);
            }
        }
    }
    
    // Dibujar información del juego
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(10, 30);
    string turnText = "Turno: " + (currentPlayer == 0 ? "Jugador (NEGRAS)" : "Computadora (ROJAS)");
    for (char c : turnText) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }
    
    glRasterPos2f(10, 60);
    string diffText = "Dificultad: " + to_string(difficulty);
    for (char c : diffText) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    }
    
    glRasterPos2f(10, 80);
    string controlsText = "Click: Seleccionar/Mover | R: Reiniciar";
    for (char c : controlsText) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
    }
    
    glutSwapBuffers();
}

// Función para manejar clicks del mouse
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        int row, col;
        mouseToBoard(x, y, row, col);
        
        cout << "Click en: x=" << x << ", y=" << y << " -> Tablero: row=" << row << ", col=" << col << endl;
        
        if (currentPlayer == 0) { // Turno del jugador humano
            if (!isSelected) {
                // Seleccionar ficha
                if (gameBoard.board[row][col] == 1) { // Ficha negra
                    selectedX = col;
                    selectedY = row;
                    isSelected = true;
                    cout << "Ficha seleccionada en (" << row << "," << col << ")" << endl;
                } else {
                    cout << "No es una ficha del jugador o casilla vacía" << endl;
                }
            } else {
                // Mover ficha
                if (gameBoard.board[row][col] == 1) { // Otra ficha del jugador
                    // Cambiar selección
                    selectedX = col;
                    selectedY = row;
                    cout << "Nueva ficha seleccionada en (" << row << "," << col << ")" << endl;
                } else if (gameBoard.board[row][col] == 0) { // Casilla vacía
                    cout << "Intentando mover de (" << selectedY << "," << selectedX << ") a (" << row << "," << col << ")" << endl;
                    // Usar la función correcta de CBoard
                    if (gameBoard.movePiece(selectedY, selectedX, row, col)) {
                        cout << "Movimiento válido" << endl;
                        isSelected = false;
                        selectedX = -1;
                        selectedY = -1;
                        currentPlayer = 1 - currentPlayer; // Cambiar turno
                        
                        // Si es turno de la computadora, hacer movimiento automático
                        if (currentPlayer == 1) {
                            // Generar árbol de movimientos
                            gameTree.generateTree(gameBoard, 2); // 2 = fichas rojas
                            
                            // Obtener el mejor movimiento
                            CTree::Node* bestMove = gameTree.getBestMove();
                            if (bestMove) {
                                // Aplicar el movimiento
                                gameBoard = bestMove->board;
                                cout << "Computadora movió de (" << bestMove->fromRow << "," << bestMove->fromCol << ") a (" << bestMove->toRow << "," << bestMove->toCol << ")" << endl;
                            }
                            
                            // Cambiar de vuelta al jugador
                            currentPlayer = 0;
                        }
                    } else {
                        cout << "Movimiento inválido" << endl;
                    }
                } else {
                    cout << "No se puede mover a esa casilla" << endl;
                }
            }
        }
        
        // Actualizar pantalla
        glutPostRedisplay();
    }
}

// Función para manejar teclas
void keyboard(unsigned char key, int x, int y) {
    if (key == 'r' || key == 'R') {
        // Reiniciar juego
        gameBoard.initializeBoard();
        currentPlayer = 0;
        isSelected = false;
        selectedX = -1;
        selectedY = -1;
        cout << "Juego reiniciado" << endl;
        glutPostRedisplay();
    }
}

// Función de display
void display() {
    drawBoard();
}

// Función para configurar la dificultad
void setDifficulty(int diff) {
    difficulty = diff;
    cout << "Dificultad configurada al nivel: " << difficulty << endl;
}

// Función para inicializar OpenGL
void initGL() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 800, 600, 0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

#endif