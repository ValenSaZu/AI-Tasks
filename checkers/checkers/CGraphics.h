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
int selectedX = -1, selectedY = -1; // Casilla seleccionada
int difficulty = 3; // Nivel de dificultad (se setear� por consola)
bool gameOver = false;
int winner = -1;

// Dimensiones de la ventana y del tablero
const int WINDOW_SIZE = 600;
const int BOARD_SIZE = 8;
const int CELL_SIZE = WINDOW_SIZE / BOARD_SIZE;

// Colores
const float LIGHT_SQUARE_COLOR[] = { 0.96f, 0.87f, 0.70f }; // Beige claro
const float DARK_SQUARE_COLOR[] = { 0.54f, 0.27f, 0.07f };  // Marr�n
const float BLACK_PIECE_COLOR[] = { 0.2f, 0.2f, 0.2f };     // Negro
const float RED_PIECE_COLOR[] = { 0.8f, 0.2f, 0.2f };       // Rojo
const float SELECTED_COLOR[] = { 0.0f, 1.0f, 0.0f, 0.3f };  // Verde transparente

// Funci�n para setear la dificultad desde main.cpp
void setDifficulty(int level) {
    difficulty = level;
}

// Dibuja un cuadrado en la posici�n especificada
void drawSquare(int x, int y, const float color[]) {
    glColor3fv(color);
    glBegin(GL_QUADS);
    glVertex2f(x * CELL_SIZE, y * CELL_SIZE);
    glVertex2f((x + 1) * CELL_SIZE, y * CELL_SIZE);
    glVertex2f((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE);
    glVertex2f(x * CELL_SIZE, (y + 1) * CELL_SIZE);
    glEnd();
}

// Dibuja un c�rculo en la posici�n especificada
void drawCircle(float x, float y, float radius, const float color[]) {
    glColor3fv(color);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x, y); // Centro del c�rculo
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

// Dibuja la selecci�n actual
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
    // Dibujar informaci�n del juego
    int blackCount, redCount;
    gameBoard.countPieces(blackCount, redCount);

    string statusText;
    if (gameOver) {
        if (winner == 0) statusText = "�Ganaron las negras!";
        else if (winner == 1) statusText = "�Ganaron las rojas!";
        else statusText = "�Empate!";
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

void printBoardDebug() {
    cout << "\n=== ESTADO DEL TABLERO ===" << endl;
    cout << "Turno actual: " << (currentPlayer == 0 ? "Jugador (NEGRAS)" : "Computadora (ROJAS)") << endl;
    
    // Imprimir números de columnas
    cout << "  ";
    for (int j = 0; j < BOARD_SIZE; j++) {
        cout << j << " ";
    }
    cout << endl;
    
    // Imprimir tablero con números de filas
    for (int i = 0; i < BOARD_SIZE; i++) {
        cout << i << " ";
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (i == selectedX && j == selectedY) {
                cout << "S "; // Casilla seleccionada
            } else {
                cout << gameBoard.board[i][j] << " ";
            }
        }
        cout << endl;
    }
    cout << "0=vacío, 1=negra, 2=roja, S=seleccionada" << endl;
    cout << "==========================\n" << endl;
}

// Funci�n de visualizaci�n principal
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    drawBoard();
    drawSelection();
    drawPieces();

    printBoardDebug();

    // Cambiar la matriz de proyecci�n para dibujar texto
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

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !gameOver && currentPlayer == 0) {
        // Convertir coordenadas de pantalla a coordenadas del tablero
        int col = x / CELL_SIZE;  // j en el tablero
        int row = BOARD_SIZE - 1 - (y / CELL_SIZE);  // i en el tablero
        
        cout << "Click en: x=" << x << ", y=" << y 
             << " -> Tablero: row=" << row << ", col=" << col << endl;

        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            // Si ya hay una ficha seleccionada, intentar moverla
            if (selectedX >= 0 && selectedY >= 0) {
                cout << "Intentando mover de (" << selectedX << "," << selectedY 
                     << ") a (" << row << "," << col << ")" << endl;
                
                if (gameBoard.movePiece(selectedX, selectedY, row, col, currentPlayer)) {
                    cout << "Movimiento exitoso!" << endl;
                    currentPlayer = 1; // Cambiar a turno de la computadora
                    
                    // Verificar si el juego ha terminado
                    if (gameBoard.isGameOver()) {
                        gameOver = true;
                        winner = gameBoard.getWinner();
                        cout << "Juego terminado! Ganador: " << winner << endl;
                    }
                } else {
                    cout << "Movimiento inválido" << endl;
                }
                selectedX = -1;
                selectedY = -1;
            }
            // Seleccionar una ficha del jugador actual (NEGRAS = 1)
            else if (gameBoard.board[row][col] == 1) {
                cout << "Ficha seleccionada en (" << row << "," << col << ")" << endl;
                selectedX = row;
                selectedY = col;
            }
            else {
                cout << "No es una ficha del jugador o casilla vacía" << endl;
            }
        }
        glutPostRedisplay();
    }
}

// Realiza el movimiento de la computadora
void computerMove() {
    if (currentPlayer == 1 && !gameOver) {
        cout << "Computadora pensando..." << endl;
        
        int startX, startY, endX, endY;
        gameTree.setMaxDepth(difficulty);
        
        if (gameTree.findBestMove(gameBoard, startX, startY, endX, endY, true)) {
            cout << "Computadora mueve de (" << startX << "," << startY 
                 << ") a (" << endX << "," << endY << ")" << endl;
            
            gameBoard.movePiece(startX, startY, endX, endY, currentPlayer);
            currentPlayer = 0; // Volver al turno del jugador

            // Verificar si el juego ha terminado
            if (gameBoard.isGameOver()) {
                gameOver = true;
                winner = gameBoard.getWinner();
                cout << "Juego terminado! Ganador: " << winner << endl;
            }
        } else {
            cout << "Computadora no encontró movimientos válidos" << endl;
            currentPlayer = 0; // Pasar turno si no hay movimientos
        }
        
        glutPostRedisplay();
    }
}

// Temporizador para el movimiento de la computadora
void timer(int value) {
    if (currentPlayer == 1 && !gameOver) {
        cout << "Timer activado - Turno de la computadora" << endl;
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