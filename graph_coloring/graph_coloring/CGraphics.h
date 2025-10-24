#ifndef CGRAPHICS_H
#define CGRAPHICS_H

#include "CGraph.h"
#include "CColoring.h"
#include <vector>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <GL/glut.h>
#include <string>
#include <sstream>

using namespace std;

class CGraphics {
public:
    CGraphics(int windowWidth = 1000, int windowHeight = 800, int nodeRadius = 20)
        : windowWidth(windowWidth), windowHeight(windowHeight),
        baseNodeRadius(nodeRadius), selectedAlgorithm(0), nColors(4) {
        graph = new Graph(8); // Grafo por defecto con 8 nodos
        calculateNodePositions();
    }

    ~CGraphics() {
        if (graph) delete graph;
    }

    void initializeGraph(int numNodes = 8, double density = 0.57) {
        if (graph) delete graph;
        graph = new Graph(numNodes, density);
        calculateNodePositions();

        if (onGraphChanged) onGraphChanged();
    }

    void drawGraph() const {
        if (!graph) return;

        float dynamicRadius = getDynamicNodeRadius();

        // Dibujar conexiones
        glColor3f(0.5, 0.5, 0.5);
        glLineWidth(1.5);

        for (const auto& node : graph->nodes) {
            float startX, startY;
            gridToScreen(node.x, node.y, startX, startY);

            for (const auto& edge : node.neighbors) {
                float endX, endY;
                gridToScreen(graph->nodes[edge.destination].x,
                    graph->nodes[edge.destination].y, endX, endY);

                glBegin(GL_LINES);
                glVertex2f(startX, startY);
                glVertex2f(endX, endY);
                glEnd();
            }
        }

        // Dibujar nodos
        for (const auto& node : graph->nodes) {
            float screenX, screenY;
            gridToScreen(node.x, node.y, screenX, screenY);

            // Asignar color según el color del nodo
            switch (node.color) {
            case 0: glColor3f(1.0, 0.0, 0.0); break; // Rojo
            case 1: glColor3f(0.0, 1.0, 0.0); break; // Verde
            case 2: glColor3f(0.0, 0.0, 1.0); break; // Azul
            case 3: glColor3f(1.0, 1.0, 0.0); break; // Amarillo
            case 4: glColor3f(1.0, 0.0, 1.0); break; // Magenta
            case 5: glColor3f(0.0, 1.0, 1.0); break; // Cian
            case 6: glColor3f(1.0, 0.5, 0.0); break; // Naranja
            case 7: glColor3f(0.5, 0.0, 0.5); break; // Púrpura
            case 8: glColor3f(0.5, 0.5, 0.5); break; // Gris
            case 9: glColor3f(0.0, 0.5, 0.5); break; // Verde azulado
            default: glColor3f(0.8, 0.8, 0.8); break; // Gris (sin color)
            }

            // Dibujar círculo del nodo
            glBegin(GL_TRIANGLE_FAN);
            glVertex2f(screenX, screenY);
            for (int i = 0; i <= 360; i += 10) {
                float angle = i * 3.14159 / 180;
                glVertex2f(screenX + dynamicRadius * cos(angle),
                    screenY + dynamicRadius * sin(angle));
            }
            glEnd();

            // Borde del nodo
            glColor3f(0.0, 0.0, 0.0);
            glLineWidth(2.0);
            glBegin(GL_LINE_LOOP);
            for (int i = 0; i <= 360; i += 10) {
                float angle = i * 3.14159 / 180;
                glVertex2f(screenX + dynamicRadius * cos(angle),
                    screenY + dynamicRadius * sin(angle));
            }
            glEnd();

            // Dibujar ID del nodo
            glColor3f(0.0, 0.0, 0.0);
            glRasterPos2f(screenX - 5, screenY - 5);
            string idStr = to_string(node.id);
            for (char c : idStr) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
            }

            // Dibujar color asignado
            glRasterPos2f(screenX - 5, screenY - 20);
            string colorStr = "C:" + to_string(node.color);
            for (char c : colorStr) {
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c);
            }
        }

        // Dibujar información del algoritmo seleccionado
        drawAlgorithmInfo();
    }

    void handleKeyboard(unsigned char key) {
        if (!graph) return;

        switch (key) {
        case '1':
            selectedAlgorithm = 0;
            cout << "Ejecutando: Busqueda Ciega" << endl;
            BlindSearch(*graph, nColors);
            break;
        case '2':
            selectedAlgorithm = 1;
            cout << "Ejecutando: Backtracking Busqueda Ciega" << endl;
            BacktrackingBlindSearch(*graph, nColors);
            break;
        case '3':
            selectedAlgorithm = 2;
            cout << "Ejecutando: Forward Checking" << endl;
            ForwardChecking(*graph, nColors);
            break;
        case '4':
            selectedAlgorithm = 3;
            cout << "Ejecutando: Variable Mas Restrictiva" << endl;
            variableMoreRestrictive(*graph, nColors);
            break;
        case '5':
            selectedAlgorithm = 4;
            cout << "Ejecutando: Variable Mas Restringida" << endl;
            variableMoreRestricted(*graph, nColors);
            break;
        case '6':
            selectedAlgorithm = 5;
            cout << "Ejecutando: Valor Menos Restrictivo" << endl;
            valueLessRestrictive(*graph, nColors);
            break;
        case 'r':
            cout << "Reiniciando colores..." << endl;
            graph->cleanAll();
            break;
        case 'n':
            cout << "Generando nuevo grafo..." << endl;
            initializeGraph(8, 0.57);
            break;
        case '+':
            nColors = min(10, nColors + 1);
            cout << "Numero de colores: " << nColors << endl;
            break;
        case '-':
            nColors = max(2, nColors - 1);
            cout << "Numero de colores: " << nColors << endl;
            break;
        }
        if (onAlgorithmChanged) onAlgorithmChanged();
    }

    int getSelectedAlgorithm() const { return selectedAlgorithm; }
    Graph* getGraph() const { return graph; }
    int getNColors() const { return nColors; }

    // Callbacks para interacción externa
    function<void()> onGraphChanged;
    function<void()> onAlgorithmChanged;

private:
    Graph* graph = nullptr;
    int windowWidth, windowHeight;
    int baseNodeRadius;
    int selectedAlgorithm;
    int nColors;

    float getDynamicNodeRadius() const {
        if (!graph || graph->nodes.empty()) return baseNodeRadius;

        // Ajustar radio según número de nodos
        float dynamicRadius = 200.0f / sqrt(graph->nodes.size());
        dynamicRadius = max(10.0f, min(static_cast<float>(baseNodeRadius), dynamicRadius));

        return dynamicRadius;
    }

    void gridToScreen(float gridX, float gridY, float& screenX, float& screenY) const {
        // Convertir coordenadas normalizadas (0-1) a coordenadas de pantalla
        screenX = gridX * windowWidth;
        screenY = (1.0f - gridY) * windowHeight; // Invertir Y para coordenadas GLUT
    }

    void calculateNodePositions() {
        if (!graph) return;

        int n = graph->nodes.size();
        float radius = 0.4f;
        float angle = 2.0f * 3.14159f / n;

        for (int i = 0; i < n; i++) {
            graph->nodes[i].x = 0.5f + radius * cos(i * angle);
            graph->nodes[i].y = 0.5f + radius * sin(i * angle);
        }
    }

    void drawAlgorithmInfo() const {
        vector<string> algorithmNames = {
            "1. Busqueda Ciega",
            "2. Backtracking Busqueda Ciega",
            "3. Forward Checking",
            "4. Variable Mas Restrictiva",
            "5. Variable Mas Restringida",
            "6. Valor Menos Restrictivo"
        };

        glColor3f(0.0, 0.0, 0.0);
        glRasterPos2f(10, windowHeight - 20);
        string info = "Algoritmo: " + algorithmNames[selectedAlgorithm];
        for (char c : info) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
        }

        // Información de colores
        glRasterPos2f(10, windowHeight - 40);
        string colorsInfo = "Colores: " + to_string(nColors) + " (+/- para cambiar)";
        for (char c : colorsInfo) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
        }

        // Instrucciones
        glRasterPos2f(10, windowHeight - 60);
        string instructions = "Teclas: 1-6 (algoritmos), r (reiniciar), n (nuevo grafo)";
        for (char c : instructions) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
        }
    }
};

#endif