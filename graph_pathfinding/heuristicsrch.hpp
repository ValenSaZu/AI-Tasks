/*
============================================================
 Explicación del código

 Estructura usada:
 - Grafo modelado como una grilla (Graph en graph.hpp).
   Cada nodo tiene:
      id    -> índice único
      (x,y) -> coordenadas
      heuristic -> distancia Chebyshev al objetivo (peso 1 para todos los caminos)
      neighbors -> lista de aristas con peso (Edge)

 Archivos:
 - graph.hpp        -> definición de Node, Edge y Graph
                       incluye generación de grillas y heurísticas.
 - heuristicsrch.hpp -> algoritmos heurísticos:
                        * Hill Climbing
                        * Best-First Search
                        * A* Search

 Notas:
 - El grafo se genera con Graph::buildGrid(width, height). -> No lo puse en un constructor para mayor flexibilidad (puedes borrar 
                                                            el grid anterior siempre que requieras sin crear un graph nuevo)
 - Luego se asignan heurísticas con Graph::setHeuristics(goalId). -> Global para todos los algoritmos listados
 - Los algoritmos devuelven un vector<int> con la secuencia de nodos. -> Facilidad de pintar el camino
============================================================
*/



#ifndef HEURISTICSRCH_HPP
#define HEURISTICSRCH_HPP

#include <queue>
#include <vector>
#include <unordered_set>
#include <limits>
#include "graph.hpp"

// Hill climbing search

vector<int> hillClimbing(Graph& g, int start, int goal) {
    vector<int> path;
    int current = start;
    path.push_back(current);

    while (current != goal) {
        int bestNeighbor = -1;
        int bestHeuristic = g.nodes[current].heuristic;

        for (auto& e : g.nodes[current].neighbors) {
            int h = g.nodes[e.destination].heuristic;
            if (h < bestHeuristic) {
                bestHeuristic = h;
                bestNeighbor = e.destination;
            }
        }

        if (bestNeighbor == -1) {
            // si no hay mejor camino (atrapado)
            break;
        }

        current = bestNeighbor;
        path.push_back(current);
    }

    return path;
}


// Best-first search

struct CompareH {
    bool operator()(const Node* a, const Node* b) const {
        return a->heuristic > b->heuristic; // menor heurística tiene prioridad
    }
};

std::vector<int> bestFirstSearch(Graph& g, int startId, int goalId) {
    std::priority_queue<Node*, std::vector<Node*>, CompareH> open;
    std::unordered_set<int> visited;
    std::vector<int> parent(g.nodes.size(), -1);

    Node* start = &g.nodes[startId];
    Node* goal = &g.nodes[goalId];

    open.push(start);

    while (!open.empty()) {
        Node* current = open.top();
        open.pop();

        if (current->id == goal->id) {
            // reconstruir camino
            std::vector<int> path;
            for (int at = goal->id; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        if (visited.count(current->id)) continue;
        visited.insert(current->id);

        for (auto& e : current->neighbors) {
            if (!visited.count(e.destination)) {
                parent[e.destination] = current->id;
                open.push(&g.nodes[e.destination]);
            }
        }
    }

    return {}; // sin camino
}

// A* search

struct CompareF {
    bool operator()(const std::pair<Node*, int>& a, const std::pair<Node*, int>& b) const {
        return a.second > b.second; // menor f tiene prioridad
    }
};

std::vector<int> aStar(Graph& g, int startId, int goalId) {
    int n = g.nodes.size();
    std::vector<int> gScore(n, std::numeric_limits<int>::max());
    std::vector<int> parent(n, -1);

    Node* start = &g.nodes[startId];
    Node* goal = &g.nodes[goalId];

    gScore[start->id] = 0;

    std::priority_queue<
        std::pair<Node*, int>,
        std::vector<std::pair<Node*, int>>,
        CompareF
    > open;

    open.push({ start, start->heuristic }); // f = g + h = 0 + h

    while (!open.empty()) {
        Node* current = open.top().first;
        open.pop();

        if (current->id == goal->id) {
            // reconstruir camino
            std::vector<int> path;
            for (int at = goal->id; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (auto& e : current->neighbors) {
            int tentative_g = gScore[current->id] + e.weight;

            if (tentative_g < gScore[e.destination]) {
                parent[e.destination] = current->id;
                gScore[e.destination] = tentative_g;
                int f = tentative_g + g.nodes[e.destination].heuristic;
                open.push({ &g.nodes[e.destination], f });
            }
        }
    }

    return {}; // sin camino
}

#endif