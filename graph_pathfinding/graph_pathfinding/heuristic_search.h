#ifndef HEURISTIC_SEARCH_HPP
#define HEURISTIC_SEARCH_HPP

#include "graph.h"
#include "search_result.h"
#include <queue>
#include <vector>
#include <unordered_set>
#include <limits>
#include <algorithm>
using namespace std;

// Hill climbing search
SearchResult hillClimbing(Graph& g, int start, int goal) {
    vector<int> path;
    vector<int> explored;
    int current = start;
    path.push_back(current);
    explored.push_back(current);

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
        explored.push_back(current);
    }

    return { path, explored };
}

// Best-first search
struct CompareH {
    bool operator()(const Node* a, const Node* b) const {
        return a->heuristic > b->heuristic; // menor heurística tiene prioridad
    }
};

SearchResult bestFirstSearch(Graph& g, int startId, int goalId) {
    priority_queue<Node*, vector<Node*>, CompareH> open;
    unordered_set<int> visited;
    vector<int> parent(g.nodes.size(), -1);
    vector<int> explored; // Para guardar el orden de exploración

    Node* start = &g.nodes[startId];
    Node* goal = &g.nodes[goalId];

    open.push(start);

    while (!open.empty()) {
        Node* current = open.top();
        open.pop();

        if (visited.count(current->id)) continue;
        visited.insert(current->id);
        explored.push_back(current->id); // Registrar nodo explorado

        if (current->id == goal->id) {
            // reconstruir camino
            vector<int> path;
            for (int at = goal->id; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            reverse(path.begin(), path.end());
            return { path, explored };
        }

        for (auto& e : current->neighbors) {
            if (!visited.count(e.destination)) {
                parent[e.destination] = current->id;
                open.push(&g.nodes[e.destination]);
            }
        }
    }

    return { {}, explored }; // sin camino
}

// A* search
struct CompareF {
    bool operator()(const pair<Node*, int>& a, const pair<Node*, int>& b) const {
        return a.second > b.second; // menor f tiene prioridad
    }
};

SearchResult aStar(Graph& g, int startId, int goalId) {
    int n = g.nodes.size();
    vector<int> gScore(n, numeric_limits<int>::max());
    vector<int> parent(n, -1);
    vector<int> explored; // Para guardar el orden de exploración
    unordered_set<int> closedSet;

    Node* start = &g.nodes[startId];
    Node* goal = &g.nodes[goalId];

    gScore[start->id] = 0;

    priority_queue<
        pair<Node*, int>,
        vector<pair<Node*, int>>,
        CompareF
    > open;

    open.push({ start, start->heuristic }); // f = g + h = 0 + h

    while (!open.empty()) {
        Node* current = open.top().first;
        open.pop();

        if (closedSet.count(current->id)) continue;
        closedSet.insert(current->id);
        explored.push_back(current->id); // Registrar nodo explorado

        if (current->id == goal->id) {
            // reconstruir camino
            vector<int> path;
            for (int at = goal->id; at != -1; at = parent[at]) {
                path.push_back(at);
            }
            reverse(path.begin(), path.end());
            return { path, explored };
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

    return { {}, explored }; // sin camino
}

#endif