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
int currentPlayer = 0; // 0: jugador (negras), 1: computadora (rojas)
int selectedX = -1, selectedY = -1; // Casilla seleccionada
int difficulty = 3; // Nivel de dificultad (se seteará por consola)
bool gameOver = false;
int winner = -1;

// Dimensiones de la ventana y del tablero
const int WINDOW_SIZE = 600;
const int BOARD_SIZE = 8;
const int CELL_SIZE = WINDOW_SIZE / BOARD_SIZE;

// Colores
const float LIGHT_SQUARE_COLOR[] = { 0.96f, 0.87f, 0.70f }; // Beige claro
const float DARK_SQUARE_COLOR[] = { 0.54f, 0.27f, 0.07f };  // Marrón
const float BLACK_PIECE_COLOR[] = { 0.2f, 0.2f, 0.2f };     // Negro
const float RED_PIECE_COLOR[] = { 0.8f, 0.2f, 0.2f };       // Rojo
const float SELECTED_COLOR[] = { 0.0f, 1.0f, 0.0f, 0.3f };  // Verde transparente

// Función para setear la dificultad desde main.cpp
void setDifficulty(int level) {
    difficulty = level;
}

// Dibuja un cuadrado en la posición especificada
void drawSquare(int x, int y, const float color[]) {
    glColor3fv(color);
    glBegin(GL_QUADS);
    glVertex2f(x * CELL_SIZE, y * CELL_SIZE);
    glVertex2f((x + 1) * CELL_SIZE, y * CELL_SIZE);
    glVertex2f((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE);
    glVertex2f(x * CELL_SIZE, (y + 1) * CELL_SIZE);
    glEnd();
}

// Dibuja un círculo en la posición especificada
void drawCircle(float x, float y, float radius, const float color[]) {
    glColor3fv(color);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y); // Centro del círculo
    for (int i = 0; i <= 360; i += 10) {
        float angle = i * 3.14159f / 180.0f;
        glVertex2f(x + radius * cos(angle), y + radius * sin(angle));
    }
    glEnd();
}

// Dibuja el tablero de damas
void drawBoard() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if ((i + j) % 2 == 0) {
                drawSquare(j, BOARD_SIZE - 1 - i, LIGHT_SQUARE_COLOR);
            }
            else {
                drawSquare(j, BOARD_SIZE - 1 - i, DARK_SQUARE_COLOR);
            }
        }
    }
}

// Dibuja las fichas en el tablero
void drawPieces() {
    float radius = CELL_SIZE * 0.4f;

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            int piece = gameBoard.board[i][j];
            if (piece != 0) {
                float x = j * CELL_SIZE + CELL_SIZE / 2.0f;
                float y = (BOARD_SIZE - 1 - i) * CELL_SIZE + CELL_SIZE / 2.0f;

                if (piece == 1) { // Ficha negra
                    drawCircle(x, y, radius, BLACK_PIECE_COLOR);
                }
                else if (piece == 2) { // Ficha roja
                    drawCircle(x, y, radius, RED_PIECE_COLOR);
                }
            }
        }
    }
}

// Dibuja la selección actual
void drawSelection() {
    if (selectedX >= 0 && selectedY >= 0) {
        glColor4fv(SELECTED_COLOR);
        glBegin(GL_QUADS);
        glVertex2f(selectedY * CELL_SIZE, (BOARD_SIZE - 1 - selectedX) * CELL_SIZE);
        glVertex2f((selectedY + 1) * CELL_SIZE, (BOARD_SIZE - 1 - selectedX) * CELL_SIZE);
        glVertex2f((selectedY + 1) * CELL_SIZE, (BOARD_SIZE - selectedX) * CELL_SIZE);
        glVertex2f(selectedY * CELL_SIZE, (BOARD_SIZE - selectedX) * CELL_SIZE);
        glEnd();
    }
}

// Dibuja texto en la pantalla
void drawText(float x, float y, const char* text) {
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2f(x, y);
    for (const char* c = text; *c != '\0'; c++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }
}

// Dibuja la interfaz de usuario
void drawUI() {
    // Dibujar información del juego
    int blackCount, redCount;
    gameBoard.countPieces(blackCount, redCount);

    string statusText;
    if (gameOver) {
        if (winner == 0) statusText = "¡Ganaron las negras!";
        else if (winner == 1) statusText = "¡Ganaron las rojas!";
        else statusText = "¡Empate!";
    }
    else {
        statusText = currentPlayer == 0 ? "Turno: Jugador (Negras)" : "Turno: Computadora (Rojas)";
    }

    drawText(10, WINDOW_SIZE + 20, statusText.c_str());

    string piecesText = "Fichas: Negras=" + to_string(blackCount) + " Rojas=" + to_string(redCount);
    drawText(10, WINDOW_SIZE + 40, piecesText.c_str());

    string difficultyText = "Dificultad: Nivel " + to_string(difficulty);
    drawText(10, WINDOW_SIZE + 60, difficultyText.c_str());

    if (gameOver) {
        drawText(WINDOW_SIZE / 2 - 50, WINDOW_SIZE + 40, "Presiona R para reiniciar");
    }
}

// Función de visualización principal
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    drawBoard();
    drawSelection();
    drawPieces();

    // Cambiar la matriz de proyección para dibujar texto
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, WINDOW_SIZE, 0, WINDOW_SIZE + 80);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    drawUI();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glutSwapBuffers();
}

// Maneja el clic del mouse
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !gameOver && currentPlayer == 0) {
        int boardY = x / CELL_SIZE;
        int boardX = BOARD_SIZE - 1 - (y / CELL_SIZE);

        if (boardX >= 0 && boardX < BOARD_SIZE && boardY >= 0 && boardY < BOARD_SIZE) {
            // Si ya hay una ficha seleccionada, intentar moverla
            if (selectedX >= 0 && selectedY >= 0) {
                if (gameBoard.movePiece(selectedX, selectedY, boardX, boardY, currentPlayer)) {
                    currentPlayer = 1 - currentPlayer; // Cambiar turno

                    // Verificar si el juego ha terminado
                    if (gameBoard.isGameOver()) {
                        gameOver = true;
                        winner = gameBoard.getWinner();
                    }
                }
                selectedX = -1;
                selectedY = -1;
            }
            // Seleccionar una ficha del jugador actual
            else if (gameBoard.board[boardX][boardY] == 1) {
                selectedX = boardX;
                selectedY = boardY;
            }
        }
        glutPostRedisplay();
    }
}

// Realiza el movimiento de la computadora
void computerMove() {
    if (currentPlayer == 1 && !gameOver) {
        int startX, startY, endX, endY;

        gameTree.setMaxDepth(difficulty);
        if (gameTree.findBestMove(gameBoard, startX, startY, endX, endY, true)) {
            gameBoard.movePiece(startX, startY, endX, endY, currentPlayer);
        }

        currentPlayer = 0; // Volver al turno del jugador

        // Verificar si el juego ha terminado
        if (gameBoard.isGameOver()) {
            gameOver = true;
            winner = gameBoard.getWinner();
        }

        glutPostRedisplay();
    }
}

// Temporizador para el movimiento de la computadora
void timer(int value) {
    if (currentPlayer == 1 && !gameOver) {
        computerMove();
    }
    glutTimerFunc(1000, timer, 0); // Llamar cada segundo
}

// Maneja las teclas normales
void keyboard(unsigned char key, int x, int y) {
    // Reiniciar juego con la tecla R
    if (key == 'r' || key == 'R') {
        gameBoard.initializeBoard();
        currentPlayer = 0;
        selectedX = -1;
        selectedY = -1;
        gameOver = false;
        winner = -1;
        glutPostRedisplay();
    }
}

// Inicializa OpenGL
void initGL() {
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WINDOW_SIZE, 0, WINDOW_SIZE);
    glMatrixMode(GL_MODELVIEW);
}

#endif /* CGraphics_h */