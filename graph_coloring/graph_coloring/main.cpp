#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <Windows.h>
#include <GL/glut.h>
#include <iostream>
#include "CGraphics.h"

using namespace std;

// Constantes
const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 800;

// Variables globales
CGraphics visualizer(WINDOW_WIDTH, WINDOW_HEIGHT, 20);

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    visualizer.drawGraph();
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
    visualizer.handleKeyboard(key);
    glutPostRedisplay();

    if (key == 27) exit(0); // ESC para salir
}

void init() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT);
}

int main(int argc, char** argv) {
    // Mostrar la consola para entrada/salida
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Graph Coloring Algorithms");

    init();
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    cout << "Controles:" << endl;
    cout << "1: Busqueda Ciega" << endl;
    cout << "2: Backtracking Busqueda Ciega" << endl;
    cout << "3: Forward Checking" << endl;
    cout << "4: Variable Mas Restrictiva" << endl;
    cout << "5: Variable Mas Restringida" << endl;
    cout << "6: Valor Menos Restrictivo" << endl;
    cout << "r: Reiniciar colores" << endl;
    cout << "n: Nuevo grafo" << endl;
    cout << "+/-: Aumentar/disminuir numero de colores" << endl;
    cout << "ESC: Salir" << endl;

    glutMainLoop();
    return 0;
}