#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;

struct Edge {
    int destination; // índice del nodo de destino
    int weight; // costo de la arista
};

struct Node {
    int id; // identificador del nodo
    int x, y; // coordenadas del nodo
    int heuristic; // valor de la heurística
    vector<Edge> neighbors; // aristas que salen de este nodo
};

class Graph {
public:
    vector<Node> nodes; // lista de nodos

    void buildGrid(int width, int height) {
        nodes.clear();
        nodes.reserve(width * height);

        // Crear nodos
        int id = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                nodes.push_back({ id++, x, y, 0, {} });
            }
        }

        // Conectar vecinos (8 direcciones)
        auto index = [width](int x, int y) { return y * width + x; };

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int from = index(x, y);

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue; // no conectarse a sí mismo
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int to = index(nx, ny);
                            nodes[from].neighbors.push_back({ to, 1 });
                        }
                    }
                }
            }
        }
    }

    // Heurística tipo Chebyshev (movimientos en 8 direcciones con costo 1)
    int chebyshevDistance(const Node& a, const Node& b) {
        return max(abs(a.x - b.x), abs(a.y - b.y));
    }

    // Asigna heurística a todos los nodos, dada la meta
    void setHeuristics(int goalId) {
        Node goalNode = nodes[goalId];
        for (auto& n : nodes) {
            n.heuristic = chebyshevDistance(n, goalNode);
        }
    }
};

#endif