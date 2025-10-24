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
    int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
    int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);

    int startX = (windowWidth - boardSize) / 2;
    int startY = (windowHeight - boardSize) / 2;

    // Convertir coordenadas del mouse a coordenadas del tablero (SIN INVERTIR)
    col = (x - startX) / (boardSize / 8);
    row = (y - startY) / (boardSize / 8);

    // Asegurar que estén dentro del rango válido (0-7)
    if (col < 0) col = 0;
    if (col >= 8) col = 7;
    if (row < 0) row = 0;
    if (row >= 8) row = 7;

    // DEBUG: Mostrar coordenadas convertidas
    cout << "Mouse: (" << x << "," << y << ") -> Tablero: (" << row << "," << col << ")" << endl;
}

// Función para dibujar el tablero
void drawBoard() {
    glClear(GL_COLOR_BUFFER_BIT);

    int boardSize = 400;
    int startX = (glutGet(GLUT_WINDOW_WIDTH) - boardSize) / 2;
    int startY = (glutGet(GLUT_WINDOW_HEIGHT) - boardSize) / 2;
    int cellSize = boardSize / 8;

    // Dibujar el tablero (SIN INVERTIR)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // Alternar colores de las casillas
            if ((i + j) % 2 == 0) {
                glColor3f(0.8f, 0.6f, 0.4f); // Color claro (beige)
            }
            else {
                glColor3f(0.4f, 0.2f, 0.0f); // Color oscuro (marrón)
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

                // Color de la ficha según el jugador
                if (gameBoard.board[i][j] == 1) {
                    glColor3f(0.0f, 0.0f, 0.0f); // Negro (jugador)
                }
                else if (gameBoard.board[i][j] == 2) {
                    glColor3f(0.8f, 0.0f, 0.0f); // Rojo (computadora)
                }

                // Dibujar círculo representando la ficha
                glBegin(GL_TRIANGLE_FAN);
                glVertex2f(centerX, centerY);
                for (int k = 0; k <= 360; k += 10) {
                    float angle = k * 3.14159f / 180.0f;
                    glVertex2f(centerX + radius * cos(angle), centerY + radius * sin(angle));
                }
                glEnd();
            }

            // Dibujar selección si existe una ficha seleccionada
            if (isSelected && selectedX == j && selectedY == i) {
                glColor3f(0.0f, 1.0f, 0.0f); // Verde para la selección
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

    // Dibujar información del juego en la parte superior
    glColor3f(1.0f, 1.0f, 1.0f); // Blanco
    glRasterPos2f(10, 30);
    string turn;
    if(currentPlayer == 0) turn = "Jugador (NEGRAS)";
	else turn = "Computadora (ROJAS)";
    string turnText = "Turno: " + turn;
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

    // Mostrar estado del juego si hay ganador
    if (gameBoard.isGameOver()) {
        int winner = gameBoard.getWinner();
        glRasterPos2f(300, 30);
        string winnerText = winner == 0 ? "¡Jugador Gana!" : winner == 1 ? "¡Computadora Gana!" : "¡Empate!";
        for (char c : winnerText) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
        }
    }

    glutSwapBuffers();
}

// Función para manejar clicks del mouse
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        int row, col;
        mouseToBoard(x, y, row, col);

        cout << "=== CLICK ===" << endl;
        cout << "Coordenadas mouse: x=" << x << ", y=" << y << endl;
        cout << "Coordenadas tablero: fila=" << row << ", columna=" << col << endl;
        cout << "Pieza en esta posición: " << gameBoard.board[row][col];
        cout << " (" << (gameBoard.board[row][col] == 1 ? "Negra" :
            gameBoard.board[row][col] == 2 ? "Roja" : "Vacía") << ")" << endl;
        cout << "Jugador actual: " << (currentPlayer == 0 ? "Humano (Negras)" : "Computadora (Rojas)") << endl;

        // Solo procesar clicks si el juego no ha terminado
        if (gameBoard.isGameOver()) {
            cout << "Juego terminado. Presiona R para reiniciar." << endl;
            return;
        }

        if (currentPlayer == 0) { // Turno del jugador humano
            if (!isSelected) {
                // Seleccionar ficha
                if (gameBoard.board[row][col] == 1) { // Ficha negra del jugador
                    selectedX = col;
                    selectedY = row;
                    isSelected = true;
                    cout << "✓ Ficha NEGRA seleccionada en (" << row << "," << col << ")" << endl;
                    cout << "Ahora selecciona una casilla VACÍA una fila ABAJO para mover" << endl;
                }
                else {
                    cout << "No es una ficha tuya. Solo puedes mover fichas NEGRAS" << endl;
                }
            }
            else {
                // Mover ficha
                if (gameBoard.board[row][col] == 1) { // Otra ficha del jugador
                    // Cambiar selección
                    selectedX = col;
                    selectedY = row;
                    cout << "✓ Nueva ficha seleccionada en (" << row << "," << col << ")" << endl;
                }
                else if (gameBoard.board[row][col] == 0) { // Casilla vacía
                    cout << "Intentando mover de (" << selectedY << "," << selectedX << ") a (" << row << "," << col << ")" << endl;

                    // DEBUG: Mostrar dirección del movimiento
                    int dx = row - selectedY;
                    int dy = col - selectedX;
                    cout << "Dirección: fila " << (dx > 0 ? "ABAJO" : "ARRIBA") << " (" << dx << "), ";
                    cout << "columna " << (dy > 0 ? "DERECHA" : "IZQUIERDA") << " (" << dy << ")" << endl;

                    // Usar la función movePiece con todos los parámetros necesarios
                    if (gameBoard.movePiece(selectedY, selectedX, row, col, currentPlayer)) {
                        cout << "✓ MOVIMIENTO VÁLIDO" << endl;
                        isSelected = false;
                        selectedX = -1;
                        selectedY = -1;
                        currentPlayer = 1; // Cambiar turno a computadora

                        // Verificar si el juego terminó después del movimiento
                        if (gameBoard.isGameOver()) {
                            int winner = gameBoard.getWinner();
                            cout << "¡Juego terminado! Ganador: " << (winner == 0 ? "Jugador" : "Computadora") << endl;
                        }
                        else {
                            // Si es turno de la computadora, hacer movimiento automático
                            cout << "Turno de la computadora..." << endl;

                            // Generar árbol de movimientos para la computadora (jugador 1 - rojas)
                            gameTree.generateTree(gameBoard, 1);

                            // Obtener el mejor movimiento
                            CNode* bestMove = gameTree.getBestMove();
                            if (bestMove != nullptr) {
                                // Aplicar el movimiento de la computadora
                                gameBoard = bestMove->board;
                                cout << "Computadora movió de (" << bestMove->fromRow << "," << bestMove->fromCol
                                    << ") a (" << bestMove->toRow << "," << bestMove->toCol << ")" << endl;

                                // Verificar si el juego terminó después del movimiento de la computadora
                                if (gameBoard.isGameOver()) {
                                    int winner = gameBoard.getWinner();
                                    cout << "¡Juego terminado! Ganador: " << (winner == 0 ? "Jugador" : "Computadora") << endl;
                                }
                                else {
                                    currentPlayer = 0; // Cambiar de vuelta al jugador humano
                                }
                            }
                            else {
                                cout << "La computadora no tiene movimientos válidos" << endl;
                                gameBoard.isGameOver();
								int winner = gameBoard.getWinner();
								cout << "¡Juego terminado! Ganador: " << (winner == 0 ? "Jugador" : "Computadora") << endl;
                            }
                        }
                    }
                    else {
                        cout << "Movimiento inválido" << endl;
                        cout << "Recuerda: Fichas negras solo pueden moverse DIAGONALMENTE HACIA ABAJO" << endl;
                    }
                }
                else {
                    cout << "No puedes mover a una casilla ocupada por el oponente" << endl;
                }
            }
        }

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

// Función de display (llamada por GLUT para redibujar la pantalla)
void display() {
    drawBoard();
}

// Función para configurar la dificultad
void setDifficulty(int diff) {
    difficulty = diff;
    // Ajustar la profundidad del árbol según la dificultad
    gameTree.setMaxDepth(difficulty);
    cout << "Dificultad configurada al nivel: " << difficulty << endl;
}

// Función para inicializar OpenGL
void initGL() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f); // Color de fondo gris oscuro
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 800, 600, 0); // Sistema de coordenadas
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

#endif