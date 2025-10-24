#include <GL/glut.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "CGraphics.h"

using namespace std;

// Constantes
const int WINDOW_WIDTH = 600;
const int WINDOW_HEIGHT = 680;

int main(int argc, char** argv) {
    
    // Pedir el nivel de dificultad al inicio
    int nivelDificultad;
    cout << "=== JUEGO DE DAMAS ===" << endl;
    cout << "Niveles de dificultad:" << endl;

    cout << "Selecciona el nivel de dificultad (1-5): ";
    cin >> nivelDificultad;

    // Setear la dificultad
    setDifficulty(nivelDificultad);

    cout << endl;
    cout << "Dificultad configurada al nivel: " << nivelDificultad << endl;
    cout << "Iniciando juego..." << endl;
    cout << endl;

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);

    // Crear título de la ventana
    string windowTitle = "Juego de Damas - Nivel " + to_string(nivelDificultad);
    glutCreateWindow(windowTitle.c_str());

    initGL();

    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(1000, timer, 0);

    cout << "=== CONTROLES ===" << endl;
    cout << "- Click izquierdo: Seleccionar y mover fichas" << endl;
    cout << "=================" << endl;

    glutMainLoop();
    return 0;
}
