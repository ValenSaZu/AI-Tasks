#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <iostream>
#include <GL/glut.h>
#include "CBoard.h"
#include "CGraphics.h"
#include "CTree.h"

using namespace std;

void initGL();
void display();
void mouse(int button, int state, int x, int y);
void keyboard(unsigned char key, int x, int y);
void setDifficulty(int diff);

int main(int argc, char** argv) {
    cout << "=== JUEGO DE DAMAS ===" << endl;
    cout << "Selecciona el nivel de dificultad: ";

    int difficulty;
    cin >> difficulty;

    cout << "Iniciando juego..." << endl;

    // Configurar la dificultad
    setDifficulty(difficulty);

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Juego de Damas");

    // Configurar OpenGL
    initGL();

    // Configurar callbacks (funciones de callback de GLUT)
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);

    // Inicializar juego
    gameBoard.initializeBoard();
    cout << "Juego de Damas iniciado" << endl;
    cout << "Controles:" << endl;
    cout << "- Click izquierdo: Seleccionar y mover fichas" << endl;
    cout << "- Tecla R: Reiniciar juego" << endl;
    cout << "Turno inicial: Jugador (fichas negras)" << endl;;

    glutMainLoop();

    return 0;
}