#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <Windows.h>
#include <GL/glut.h>
#include <iostream>
#include "search_visualizer.h"

using namespace std;

// Constantes
const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 800;

// Variables globales
SearchVisualizer visualizer(WINDOW_WIDTH, WINDOW_HEIGHT, 8);

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    visualizer.drawGraph();

    // Dibujar información de estado
    glColor3f(0.0, 0.0, 0.0);
    glRasterPos2f(10, WINDOW_HEIGHT - 20);

    string algoNames[] = { "BFS", "DFS", "Hill Climbing", "Best-First", "A*" };
    string status = "Algoritmo: " + algoNames[visualizer.getAlgorithmType()] +
        " | Heurística: Chebyshev";

    for (char c : status) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }

    glutSwapBuffers();
}

void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        visualizer.handleMouseClick(x, y);
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y) {
    visualizer.handleKeyboard(key);
    glutPostRedisplay();

    if (key == 27) exit(0);
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

    // Pedir el porcentaje de eliminación al inicio
    float removalPercentage;
    cout << "Porcentaje de eliminacion (0-100): ";
    cin >> removalPercentage;
    removalPercentage /= 100.0;

    // Inicializar visualizador
    visualizer.initializeGraph(100, 100, removalPercentage);


    visualizer.onGraphChanged = []() { glutPostRedisplay(); };
    visualizer.onAlgorithmChanged = []() {
        cout << "Algoritmo cambiado" << endl;
        glutPostRedisplay();
    };

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Pathfinding Algorithms - Visualización de Grafo");

    init();
    glutDisplayFunc(display);
    glutMouseFunc(mouseClick);
    glutKeyboardFunc(keyboard);

    cout << "Controles:" << endl;
    cout << "Click izquierdo: Seleccionar nodo inicio/fin" << endl;
    cout << "R: Cambiar porcentaje de eliminacion" << endl;
    cout << "1: BFS" << endl;
    cout << "2: DFS" << endl;
    cout << "3: Hill Climbing" << endl;
    cout << "4: Best-First Search" << endl;
    cout << "5: A*" << endl;
    cout << "ESC: Salir" << endl;

    glutMainLoop();
    return 0;
}