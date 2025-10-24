//
//  blind_search.h
//  BusquedasCiegas
//
//  Created by Amara Barrera on 20/08/25.
//
#ifndef BLIND_SEARCH_HPP
#define BLIND_SEARCH_HPP

#include "graph.h"
#include "search_result.h"
#include <queue>
#include <stack>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

// Búsqueda por amplitud usando el grafo común
SearchResult BFS(Graph& g, int start, int goal) {
    vector<bool> visited(g.nodes.size(), false);
    vector<int> parent(g.nodes.size(), -1);
    queue<int> q;
    vector<int> explored; // Para guardar el orden de exploración

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        explored.push_back(current); // Registrar nodo explorado

        if (current == goal) {
            // reconstruir camino
            vector<int> path;
            for (int at = goal; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            reverse(path.begin(), path.end());
            return { path, explored };
        }

        for (auto& edge : g.nodes[current].neighbors) {
            int neighbor = edge.destination;
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                parent[neighbor] = current;
                q.push(neighbor);
            }
        }
    }

    return { {}, explored }; // sin camino
}

// Búsqueda por profundidad usando el grafo común
SearchResult DFS(Graph& g, int start, int goal) {
    vector<bool> visited(g.nodes.size(), false);
    vector<int> parent(g.nodes.size(), -1);
    stack<int> s;
    vector<int> explored; // Para guardar el orden de exploración

    s.push(start);
    visited[start] = true;

    while (!s.empty()) {
        int current = s.top();
        s.pop();
        explored.push_back(current); // Registrar nodo explorado

        if (current == goal) {
            // reconstruir camino
            vector<int> path;
            for (int at = goal; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            reverse(path.begin(), path.end());
            return { path, explored };
        }

        // Para DFS, procesamos los vecinos en orden inverso para coincidir con el comportamiento tradicional
        for (int i = g.nodes[current].neighbors.size() - 1; i >= 0; i--) {
            int neighbor = g.nodes[current].neighbors[i].destination;
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                parent[neighbor] = current;
                s.push(neighbor);
            }
        }
    }

    return { {}, explored }; // sin camino
}

#endif