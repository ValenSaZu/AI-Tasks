#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <iostream>
#include <GL/glut.h>
#include "CBoard.h"
#include "CGraphics.h"

using namespace std;

int main(int argc, char** argv) {
    cout << "=== JUEGO DE DAMAS ===" << endl;
    cout << "Selecciona el nivel de dificultad: ";
    
    int difficulty;
    cin >> difficulty;
    
    if (difficulty < 1 || difficulty > 5) {
        difficulty = 3;
    }
    
    cout << "Dificultad configurada al nivel: " << difficulty << endl;
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
    
    // Configurar callbacks
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    
    // Inicializar juego
    gameBoard.initializeBoard();
    cout << "Juego de Damas iniciado" << endl;
    cout << "Controles:" << endl;
    cout << "- Click izquierdo: Seleccionar y mover fichas" << endl;
    cout << "- Tecla R: Reiniciar juego" << endl;
    
    // Iniciar loop principal
    glutMainLoop();
    
    return 0;
}