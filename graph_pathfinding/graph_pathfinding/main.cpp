#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include "graph.h"
#include "blind_search.h"
#include "heuristic_search.h"

using namespace std;

// Constantes
const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 800;
const int NODE_RADIUS = 15;

// Variables globales
Graph graph;
int gridWidth = 100;        // Tamańo fijo del grid
int gridHeight = 100;       // Tamańo fijo del grid  
float removalPercentage = 0.0;
vector<bool> removedNodes;
int startNode = -1;
int goalNode = -1;
vector<int> currentPath;
vector<int> exploredNodes;
int algorithmType = 0; // 0: BFS, 1: DFS, 2: Hill Climbing, 3: Best-First, 4: A*

// Convertir coordenadas de grid a pantalla
void gridToScreen(int gridX, int gridY, float& screenX, float& screenY) {
    float padding = 50;
    float availableWidth = WINDOW_WIDTH - 2 * padding;
    float availableHeight = WINDOW_HEIGHT - 2 * padding;

    screenX = padding + gridX * (availableWidth / (gridWidth - 1));
    screenY = padding + gridY * (availableHeight / (gridHeight - 1));
}

void initializeGraph() {
    graph.buildGrid(gridWidth, gridHeight);
    removedNodes.clear();
    removedNodes.resize(gridWidth * gridHeight, false);

    // Eliminar nodos aleatoriamente según el porcentaje
    srand(static_cast<unsigned int>(time(0)));
    int totalNodes = gridWidth * gridHeight;
    int nodesToRemove = static_cast<int>(removalPercentage * totalNodes);
    int removedCount = 0;

    // Si el porcentaje es 0, no hacer nada
    if (nodesToRemove == 0) {
        startNode = -1;
        goalNode = -1;
        currentPath.clear();
        exploredNodes.clear();
        return;
    }

    while (removedCount < nodesToRemove) {
        int nodeToRemove = rand() % totalNodes;
        if (!removedNodes[nodeToRemove]) {
            removedNodes[nodeToRemove] = true;
            removedCount++;

            // También eliminamos este nodo de los vecinos de otros nodos
            for (auto& node : graph.nodes) {
                for (auto it = node.neighbors.begin(); it != node.neighbors.end(); ) {
                    if (it->destination == nodeToRemove) {
                        it = node.neighbors.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
            }

            // Limpiar vecinos del nodo eliminado
            graph.nodes[nodeToRemove].neighbors.clear();
        }
    }

    startNode = -1;
    goalNode = -1;
    currentPath.clear();
    exploredNodes.clear();
}

void drawGraph() {
    // Dibujar conexiones primero (para que queden detrás de los nodos)
    glColor3f(0.7, 0.7, 0.7);
    glLineWidth(1.0);

    for (const auto& node : graph.nodes) {
        if (removedNodes[node.id]) continue;

        float startX, startY;
        gridToScreen(node.x, node.y, startX, startY);

        for (const auto& edge : node.neighbors) {
            // Solo dibujar cada conexión una vez (evitar duplicados)
            if (node.id < edge.destination && !removedNodes[edge.destination]) {
                float endX, endY;
                gridToScreen(graph.nodes[edge.destination].x,
                    graph.nodes[edge.destination].y, endX, endY);

                glBegin(GL_LINES);
                glVertex2f(startX, startY);
                glVertex2f(endX, endY);
                glEnd();
            }
        }
    }

    // Dibujar nodos
    for (const auto& node : graph.nodes) {
        if (removedNodes[node.id]) continue;

        float screenX, screenY;
        gridToScreen(node.x, node.y, screenX, screenY);

        // Determinar color del nodo
        if (node.id == startNode) {
            glColor3f(1.0, 0.0, 0.0); // Rojo para inicio
        }
        else if (node.id == goalNode) {
            glColor3f(0.0, 1.0, 0.0); // Verde para fin
        }
        else if (find(currentPath.begin(), currentPath.end(), node.id) != currentPath.end()) {
            glColor3f(1.0, 0.0, 0.0); // Rojo para camino final
        }
        else if (find(exploredNodes.begin(), exploredNodes.end(), node.id) != exploredNodes.end()) {
            glColor3f(0.0, 0.0, 1.0); // Azul para nodos explorados
        }
        else {
            glColor3f(0.8, 0.8, 0.8); // Gris claro para nodos normales
        }

        // Dibujar círculo
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(screenX, screenY); // centro
        for (int i = 0; i <= 360; i += 10) {
            float angle = i * 3.14159 / 180;
            glVertex2f(screenX + NODE_RADIUS * cos(angle),
                screenY + NODE_RADIUS * sin(angle));
        }
        glEnd();

        // Borde del nodo
        glColor3f(0.0, 0.0, 0.0);
        glLineWidth(2.0);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i <= 360; i += 10) {
            float angle = i * 3.14159 / 180;
            glVertex2f(screenX + NODE_RADIUS * cos(angle),
                screenY + NODE_RADIUS * sin(angle));
        }
        glEnd();
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    drawGraph();

    // Dibujar información de estado
    glColor3f(0.0, 0.0, 0.0);
    glRasterPos2f(10, WINDOW_HEIGHT - 20);

    string algoNames[] = { "BFS", "DFS", "Hill Climbing", "Best-First", "A*" };

    string status = "Algoritmo: " + algoNames[algorithmType] +
        " | Heurística: Chebyshev" +
        " | Nodos: " + to_string(gridWidth) + "x" + to_string(gridHeight) +
        " | Eliminados: " + to_string(int(removalPercentage * 100)) + "%";

    for (char c : status) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }

    glutSwapBuffers();
}

void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // Convertir coordenadas de pantalla a grid
        float padding = 50;
        float availableWidth = WINDOW_WIDTH - 2 * padding;
        float availableHeight = WINDOW_HEIGHT - 2 * padding;

        float gridX = (x - padding) * (gridWidth - 1) / availableWidth;
        float gridY = ((WINDOW_HEIGHT - y) - padding) * (gridHeight - 1) / availableHeight;

        // Encontrar el nodo más cercano
        int closestNode = -1;
        float minDist = NODE_RADIUS * 2; // Solo seleccionar si está suficientemente cerca

        for (const auto& node : graph.nodes) {
            if (removedNodes[node.id]) continue;

            float dist = sqrt(pow(node.x - gridX, 2) + pow(node.y - gridY, 2));
            if (dist < minDist) {
                minDist = dist;
                closestNode = node.id;
            }
        }

        if (closestNode != -1) {
            if (startNode == -1) {
                startNode = closestNode;
            }
            else if (goalNode == -1 && closestNode != startNode) {
                goalNode = closestNode;

                // Ejecutar algoritmo seleccionado
                SearchResult result;

                if (algorithmType >= 2) { // Algoritmos heurísticos
                    graph.setHeuristics(goalNode); // Siempre usa Chebyshev
                }

                switch (algorithmType) {
                case 0: // BFS
                    result = BFS(graph, startNode, goalNode);
                    break;
                case 1: // DFS
                    result = DFS(graph, startNode, goalNode);
                    break;
                case 2: // Hill Climbing
                    result = hillClimbing(graph, startNode, goalNode);
                    break;
                case 3: // Best-First
                    result = bestFirstSearch(graph, startNode, goalNode);
                    break;
                case 4: // A*
                    result = aStar(graph, startNode, goalNode);
                    break;
                }

                currentPath = result.path;
                exploredNodes = result.explored;
            }
            else {
                // Resetear para nueva selección
                startNode = closestNode;
                goalNode = -1;
                currentPath.clear();
                exploredNodes.clear();
            }

            glutPostRedisplay();
        }
    }
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 'r': // Cambiar solo el porcentaje de eliminación
        cout << "Nuevo porcentaje de eliminacion (0-100): ";
        cin >> removalPercentage;
        removalPercentage /= 100.0;
        initializeGraph();
        break;
    case '1':
        algorithmType = 0;
        cout << "Algoritmo: BFS" << endl;
        if (goalNode != -1) {
            SearchResult result = BFS(graph, startNode, goalNode);
            currentPath = result.path;
            exploredNodes = result.explored;
        }
        break;
    case '2':
        algorithmType = 1;
        cout << "Algoritmo: DFS" << endl;
        if (goalNode != -1) {
            SearchResult result = DFS(graph, startNode, goalNode);
            currentPath = result.path;
            exploredNodes = result.explored;
        }
        break;
    case '3':
        algorithmType = 2;
        cout << "Algoritmo: Hill Climbing" << endl;
        if (goalNode != -1) {
            graph.setHeuristics(goalNode); // Siempre usa Chebyshev
            SearchResult result = hillClimbing(graph, startNode, goalNode);
            currentPath = result.path;
            exploredNodes = result.explored;
        }
        break;
    case '4':
        algorithmType = 3;
        cout << "Algoritmo: Best-First" << endl;
        if (goalNode != -1) {
            graph.setHeuristics(goalNode); // Siempre usa Chebyshev
            SearchResult result = bestFirstSearch(graph, startNode, goalNode);
            currentPath = result.path;
            exploredNodes = result.explored;
        }
        break;
    case '5':
        algorithmType = 4;
        cout << "Algoritmo: A*" << endl;
        if (goalNode != -1) {
            graph.setHeuristics(goalNode); // Siempre usa Chebyshev
            SearchResult result = aStar(graph, startNode, goalNode);
            currentPath = result.path;
            exploredNodes = result.explored;
        }
        break;
    case 27: // ESC key
        exit(0);
        break;
    }
    glutPostRedisplay();
}

void init() {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT);
    initializeGraph();
}

int main(int argc, char** argv) {
    // Mostrar la consola para entrada/salida
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);

    // Pedir el porcentaje de eliminación al inicio
    cout << "Porcentaje de eliminacion (0-100): ";
    cin >> removalPercentage;
    removalPercentage /= 100.0;

    // Inicializar GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);

    int window = glutCreateWindow("Pathfinding Algorithms - Visualización de Grafo");

    if (window == 0) {
        cerr << "Error: No se pudo crear la ventana GLUT" << endl;
        return 1;
    }

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
    cout << "NOTA: Los algoritmos heurísticos usan siempre distancia Chebyshev" << endl;

    cout << "Entrando en glutMainLoop()..." << endl;
    glutMainLoop();

    return 0;
}