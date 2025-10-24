#ifndef COLORING_HPP
#define COLORING_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include "CGraph.h"
using namespace std;

// BUSQUEDAS CIEGAS -------------------------------------------------------------

bool isValidColor(Graph& g, int nodeIndex, int color) {
    for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
        if (g.nodes[neighbor.destination].color == color) {
            return false;
        }
    }
    return true;
}

void BlindSearch(Graph& g, int nColors) {
    cout << "Busqueda ciega -------------------------------------------------------------" << endl;
    // Inicializar colores
    for (auto& node : g.nodes) {
        node.color = 0;
    }
    // Asignar colores secuencialmente
    for (int i = 0; i < g.nodes.size(); i++) {
        // mientras haya conflicto con un vecino, prueba siguiente color
        while (true) {
            bool conflict = false;
            //verifica si el color actual genera conflicto
            conflict = isValidColor(g, i, g.nodes[i].color) == false;

            if (!conflict) break; // encontró color válido

            g.nodes[i].color++; // probar siguiente color
            if (g.nodes[i].color >= nColors) {
                cout << "No se pudo colorear el grafo con " << nColors << " colores." << endl;
                return;
            }
        }
    }

    // Si terminó, imprime la solución
    g.printColoring();
}

bool backtrack(Graph& g, int nodeIndex, int nColors) {
    if (nodeIndex == g.nodes.size()) return true; // caso base: todos coloreados

    for (int color = 0; color < nColors; color++) {
        if (isValidColor(g, nodeIndex, color)) {
            g.nodes[nodeIndex].color = color;

            if (backtrack(g, nodeIndex + 1, nColors)) return true;

            cout << "Backtracking en nodo " << nodeIndex << endl;

            g.nodes[nodeIndex].color = -1; // deshacer (backtrack), si no funciona colorear el siguiente,
            //en el for se pasara al siguiente color para el anterior y con ese nuevo color, intentar colorear el siguiente
        }
    }
    return false;
}

void BacktrackingBlindSearch(Graph& g, int nColors) {
    cout << "Backtracking Busqueda ciega -------------------------------------------------" << endl;

    // inicializar sin color
    for (auto& node : g.nodes) node.color = -1;

    if (backtrack(g, 0, nColors)) {
        g.printColoring();
    }
    else {
        cout << "No se pudo colorear con " << nColors << " colores." << endl;
    }
}

bool backtrackForward(Graph& g, int nodeIndex, int nColors) {
    if (nodeIndex == g.nodes.size()) return true; // caso base: todos coloreados
    for (int color = 0; color < nColors; color++) {
        if (isValidColor(g, nodeIndex, color)) {
            g.nodes[nodeIndex].color = color;
            // Guardar colores eliminados para deshacer si es necesario
            vector<vector<int>> removedColors(g.nodes.size());

            // Actualizar dominios de vecinos
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int i = 0; i < neighborNode.Colors.size(); i++) {
                    if (neighborNode.Colors[i] == color) {
                        removedColors[neighbor.destination].push_back(color);
                        neighborNode.Colors.erase(neighborNode.Colors.begin() + i);
                        break;
                    }
                }
            }
            // Verificar si algún vecino se quedó sin colores
            bool failure = false;
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                if (g.nodes[neighbor.destination].Colors.empty()) {
                    failure = true;
                    break;
                }
            }
            if (!failure && backtrackForward(g, nodeIndex + 1, nColors)) return true;

            cout << "Backtracking en nodo " << nodeIndex << endl;

            // Deshacer cambios en dominios
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int c : removedColors[neighbor.destination]) {
                    neighborNode.Colors.push_back(c);
                }
            }
            g.nodes[nodeIndex].color = -1; // deshacer (backtrack), esto funciona igual que en backtracking normal
        }
    }
    return false;
}

void ForwardChecking(Graph& g, int nColors) {
    cout << "Forward Checking ------------------------------------------------------------" << endl;
    // Inicializar colores y dominios
    g.addColors(nColors);

    if (backtrackForward(g, 0, nColors)) {
        g.printColoring();
    }
    else {
        cout << "No se pudo colorear con " << nColors << " colores." << endl;
    }
}

// BUSQUEDAS HEURISTICAS -------------------------------------------------------------

void variableMoreRestrictive(Graph& g, int nColors) {
    cout << "Variable mas restrictiva -----------------------------------------------------" << endl;
    // Coloca la heuristica (grado) en cada nodo y ordena de mayor a menor
    g.calculateHeuristics();
    g.sortByHeuristicDesc();

    if (backtrack(g, 0, nColors)) {
        g.printColoring();
    }
    else {
        cout << "No se pudo colorear con " << nColors << " colores." << endl;
    }
}

bool backtrackRestricted(Graph& g, int nodeIndex, int nColors) {
    if (nodeIndex == g.nodes.size()) return true; // caso base: todos coloreados
    for (auto& color : g.nodes[nodeIndex].Colors) {
        if (isValidColor(g, nodeIndex, color)) {
            g.nodes[nodeIndex].color = color;
            // Guardar colores eliminados para deshacer si es necesario
            vector<vector<int>> removedColors(g.nodes.size());
            // Actualizar dominios de vecinos
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int i = 0; i < neighborNode.Colors.size(); i++) {
                    if (neighborNode.Colors[i] == color) {
                        removedColors[neighbor.destination].push_back(color);
                        neighborNode.Colors.erase(neighborNode.Colors.begin() + i);
                        break;
                    }
                }
            }
            g.sortByAvailableColors(nodeIndex + 1); // reordena los nodos restantes por numero de colores disponibles
            if (backtrackRestricted(g, nodeIndex + 1, nColors)) return true;
            cout << "Backtracking en nodo " << nodeIndex << endl;
            // Deshacer cambios en dominios
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int c : removedColors[neighbor.destination]) {
                    neighborNode.Colors.push_back(c);
                }
            }
            g.nodes[nodeIndex].color = -1; // deshacer (backtrack), si no funciona colorear el siguiente
        }
    }
    return false;
}

void variableMoreRestricted(Graph& g, int nColors) {
    cout << "Variable mas restringida -----------------------------------------------------" << endl;
    g.addColors(nColors);

    if (backtrackRestricted(g, 0, nColors)) {
        g.printColoring();
    }
    else {
        cout << "No se pudo colorear con " << nColors << " colores." << endl;
    }
}

// Función auxiliar para calcular el "impacto" de un color (cuántos dominios afecta)
int calculateColorImpact(Graph& g, int nodeIndex, int color) {
    int impact = 0;

    // Verificar cuántos vecinos tienen este color en su dominio
    for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
        auto& neighborNode = g.nodes[neighbor.destination];
        if (find(neighborNode.Colors.begin(), neighborNode.Colors.end(), color) != neighborNode.Colors.end()) {
            impact++;
        }
    }
    return impact;
}

bool backtrackingLessRestrictive(Graph& g, int nodeIndex, int nColors) {
    if (nodeIndex == g.nodes.size()) return true; // caso base: todos coloreados

    // Ordenar colores por impacto (menos restrictivo primero)
    vector<pair<int, int>> colorImpacts; // pair<impacto, color>

    for (int color : g.nodes[nodeIndex].Colors) {
        int impact = calculateColorImpact(g, nodeIndex, color);
        colorImpacts.push_back({ impact, color });
    }

    // Ordenar por impacto ascendente (menos restrictivo primero)
    sort(colorImpacts.begin(), colorImpacts.end());

    for (auto& impactColor : colorImpacts) {
        int color = impactColor.second;

        if (isValidColor(g, nodeIndex, color)) {
            g.nodes[nodeIndex].color = color;

            // Guardar colores eliminados para deshacer
            vector<vector<int>> removedColors(g.nodes.size());

            // Actualizar dominios de vecinos
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int i = 0; i < neighborNode.Colors.size(); i++) {
                    if (neighborNode.Colors[i] == color) {
                        removedColors[neighbor.destination].push_back(color);
                        neighborNode.Colors.erase(neighborNode.Colors.begin() + i);
                        break;
                    }
                }
            }

            // Ordenar nodos restantes por MRV
            g.sortByAvailableColors(nodeIndex + 1);

            if (backtrackingLessRestrictive(g, nodeIndex + 1, nColors)) return true;

            cout << "Backtracking en nodo " << nodeIndex << " con color " << color << endl;

            // Deshacer cambios
            for (auto& neighbor : g.nodes[nodeIndex].neighbors) {
                auto& neighborNode = g.nodes[neighbor.destination];
                for (int c : removedColors[neighbor.destination]) {
                    neighborNode.Colors.push_back(c);
                }
            }
            g.nodes[nodeIndex].color = -1;
        }
    }
    return false;
}

void valueLessRestrictive(Graph& g, int nColors) {
    cout << "Valor menos restrictivo -----------------------------------------------------" << endl;
    g.addColors(nColors);

    if (backtrackingLessRestrictive(g, 0, nColors)) {
        g.printColoring();
    }
    else {
        cout << "No se pudo colorear con " << nColors << " colores." << endl;
    }
}

#endif