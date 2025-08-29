#ifndef SEARCH_VISUALIZER_HPP
#define SEARCH_VISUALIZER_HPP

#include "graph.h"
#include "blind_search.h"
#include "heuristic_search.h"
#include <vector>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>

using namespace std;

class SearchVisualizer {
public:
    SearchVisualizer(int windowWidth = 1000, int windowHeight = 800, int nodeRadius = 8)
        : windowWidth(windowWidth), windowHeight(windowHeight),
        baseNodeRadius(nodeRadius),
        startNode(-1), goalNode(-1), algorithmType(0),
        removalPercentage(0.0), gridWidth(0), gridHeight(0) {
    }

    void initializeGraph(int width, int height, float removalPct) {
        gridWidth = width;
        gridHeight = height;
        removalPercentage = removalPct;

        graph.buildGrid(gridWidth, gridHeight);
        removedNodes.clear();
        removedNodes.resize(gridWidth * gridHeight, false);

        // Eliminar nodos aleatoriamente según el porcentaje
        if (removalPercentage > 0.0) {
            srand(static_cast<unsigned int>(time(0)));
            int totalNodes = gridWidth * gridHeight;
            int nodesToRemove = static_cast<int>(removalPercentage * totalNodes);
            int removedCount = 0;

            while (removedCount < nodesToRemove) {
                int nodeToRemove = rand() % totalNodes;
                if (!removedNodes[nodeToRemove]) {
                    removedNodes[nodeToRemove] = true;
                    removedCount++;

                    // Eliminar referencias al nodo eliminado
                    for (auto& node : graph.nodes) {
                        auto it = remove_if(node.neighbors.begin(), node.neighbors.end(),
                            [nodeToRemove](const Edge& e) { return e.destination == nodeToRemove; });
                        node.neighbors.erase(it, node.neighbors.end());
                    }

                    // Limpiar vecinos del nodo eliminado
                    graph.nodes[nodeToRemove].neighbors.clear();
                }
            }
        }

        startNode = -1;
        goalNode = -1;
        currentPath.clear();
        exploredNodes.clear();

        if (onGraphChanged) onGraphChanged();
    }

    void drawGraph() const {
        float dynamicRadius = getDynamicNodeRadius();

        // Dibujar conexiones
        glColor3f(0.7, 0.7, 0.7);
        glLineWidth(1.0);

        for (const auto& node : graph.nodes) {
            if (removedNodes[node.id]) continue;

            float startX, startY;
            gridToScreen(node.x, node.y, startX, startY);

            for (const auto& edge : node.neighbors) {
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
                glColor3f(0.0, 1.0, 0.0); // Verde para inicio
            }
            else if (node.id == goalNode) {
                glColor3f(1.0, 0.0, 0.0); // Rojo para fin
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
        }
    }

    void handleMouseClick(int x, int y) {
        // Convertir coordenadas de pantalla a grid con comprobación de límites
        float padding = 50;
        float availableWidth = windowWidth - 2 * padding;
        float availableHeight = windowHeight - 2 * padding;

        if (availableWidth <= 0 || availableHeight <= 0) return;

        float gridX = (x - padding) * max(1, gridWidth - 1) / availableWidth;
        float gridY = ((windowHeight - y) - padding) * max(1, gridHeight - 1) / availableHeight;

        // Comprobar que las coordenadas están dentro del grid
        if (gridX < 0 || gridX >= gridWidth || gridY < 0 || gridY >= gridHeight) {
            return;
        }

        int closestNode = findClosestNode(gridX, gridY);
        if (closestNode == -1) return;

        if (startNode == -1) {
            startNode = closestNode;
        }
        else if (goalNode == -1 && closestNode != startNode) {
            goalNode = closestNode;
            runSelectedAlgorithm();
        }
        else {
            // Resetear para nueva selección
            startNode = closestNode;
            goalNode = -1;
            currentPath.clear();
            exploredNodes.clear();
        }
    }

    void handleKeyboard(unsigned char key) {
        switch (key) {
        case 'r':
            cout << "Nuevo porcentaje de eliminacion (0-100): ";
            float newPercentage;
            cin >> newPercentage;
            removalPercentage = newPercentage / 100.0;
            initializeGraph(gridWidth, gridHeight, removalPercentage);
            break;
        case '1': case '2': case '3': case '4': case '5':
            algorithmType = key - '1';
            if (goalNode != -1) {
                runSelectedAlgorithm();
            }
            if (onAlgorithmChanged) onAlgorithmChanged();
            break;
        }
    }

    // Getters para acceso externo
    int getAlgorithmType() const { return algorithmType; }
    int getStartNode() const { return startNode; }
    int getGoalNode() const { return goalNode; }
    const vector<int>& getCurrentPath() const { return currentPath; }
    const vector<int>& getExploredNodes() const { return exploredNodes; }
    const Graph& getGraph() const { return graph; }

    // Callbacks para interacción externa
    function<void()> onGraphChanged;
    function<void()> onAlgorithmChanged;

private:
    Graph graph;
    int gridWidth, gridHeight;
    float removalPercentage;
    vector<bool> removedNodes;
    int startNode, goalNode;
    vector<int> currentPath;
    vector<int> exploredNodes;
    int algorithmType;

    int windowWidth, windowHeight;
    int baseNodeRadius;

    float getDynamicNodeRadius() const {
        if (gridWidth <= 0 || gridHeight <= 0) return baseNodeRadius;

        // Fórmula mejorada: radio = 400 / ?(ancho * alto)
        int totalNodes = gridWidth * gridHeight;
        float dynamicRadius = 200.0f / sqrt(totalNodes);

        dynamicRadius = max(3.0f, dynamicRadius); // Mínimo 3 píxeles
        dynamicRadius = min(static_cast<float>(baseNodeRadius), dynamicRadius); // Máximo el radio base

        return dynamicRadius;
    }

    void gridToScreen(int gridX, int gridY, float& screenX, float& screenY) const {
        float padding = 50;
        float availableWidth = windowWidth - 2 * padding;
        float availableHeight = windowHeight - 2 * padding;

        float xScale = (gridWidth > 1) ? availableWidth / (gridWidth - 1) : 0;
        float yScale = (gridHeight > 1) ? availableHeight / (gridHeight - 1) : 0;

        screenX = padding + gridX * xScale;
        screenY = padding + gridY * yScale;
    }

    int findClosestNode(float gridX, float gridY) const {
        int closestNode = -1;
        float dynamicRadius = getDynamicNodeRadius();
        float minDist = dynamicRadius * 2;

        for (const auto& node : graph.nodes) {
            if (removedNodes[node.id]) continue;

            float dist = sqrt(pow(node.x - gridX, 2) + pow(node.y - gridY, 2));
            if (dist < minDist) {
                minDist = dist;
                closestNode = node.id;
            }
        }

        return closestNode;
    }

    void runSelectedAlgorithm() {
        if (startNode == -1 || goalNode == -1 ||
            !graph.nodeExists(startNode) || !graph.nodeExists(goalNode)) {
            return;
        }

        SearchResult result;

        try {
            if (algorithmType >= 2) {
                graph.setHeuristics(goalNode);
            }

            switch (algorithmType) {
            case 0: result = BFS(graph, startNode, goalNode); break;
            case 1: result = DFS(graph, startNode, goalNode); break;
            case 2: result = hillClimbing(graph, startNode, goalNode); break;
            case 3: result = bestFirstSearch(graph, startNode, goalNode); break;
            case 4: result = aStar(graph, startNode, goalNode); break;
            default: result = BFS(graph, startNode, goalNode); break;
            }

            currentPath = result.path;
            exploredNodes = result.explored;
        }
        catch (const exception& e) {
            cerr << "Error ejecutando algoritmo: " << e.what() << endl;
            currentPath.clear();
            exploredNodes.clear();
        }
    }
};

#endif