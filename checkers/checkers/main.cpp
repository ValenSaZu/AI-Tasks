//
//  main.cpp
//  ArbolDamas
//
//  Created by Amara Barrera on 4/09/25.
//

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <Windows.h>
#include <GL/glut.h>
#include <iostream>
#include <string>
#include "CGraphics.h"

using namespace std;

// Constantes
const int WINDOW_WIDTH = 600;
const int WINDOW_HEIGHT = 680; // 600 + 80 para la UI

int main(int argc, char** argv) {
    // Mostrar la consola para entrada/salida (EXACTAMENTE como en tu código)
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);

    // Pedir el nivel de dificultad al inicio (como en tu código)
    int nivelDificultad;
    cout << "=== JUEGO DE DAMAS ===" << endl;
    cout << "Niveles de dificultad:" << endl;
    cout << "1 - Facil (pensamiento a 1 movimiento)" << endl;
    cout << "2 - Intermedio (pensamiento a 2 movimientos)" << endl;
    cout << "3 - Dificil (pensamiento a 3 movimientos)" << endl;
    cout << "4 - Experto (pensamiento a 4 movimientos)" << endl;
    cout << "5 - Maestro (pensamiento a 5 movimientos)" << endl;
    cout << endl;

    cout << "Selecciona el nivel de dificultad (1-5): ";
    cin >> nivelDificultad;

    // Validar entrada
    while (nivelDificultad < 1 || nivelDificultad > 5) {
        cout << "Por favor ingresa un numero valido (1-5): ";
        cin >> nivelDificultad;
    }

    // Setear la dificultad
    setDifficulty(nivelDificultad);

    cout << endl;
    cout << "Dificultad configurada al nivel: " << nivelDificultad << endl;
    cout << "Iniciando juego..." << endl;
    cout << endl;

    // Inicializar GLUT (EXACTAMENTE como en tu código)
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
    cout << "- Flecha arriba/abajo: Cambiar dificultad durante el juego" << endl;
    cout << "- Tecla R: Reiniciar juego" << endl;
    cout << "=================" << endl;

    glutMainLoop();
    return 0;
}