#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
using namespace std;

struct Edge {
    int destination; // índice del nodo de destino
    int weight;      // costo de la arista

    Edge(int dest, int w = 1) : destination(dest), weight(w) {}
};

struct Node {
    int id;                  // identificador del nodo
    int x, y;                // coordenadas del nodo
    int heuristic;           // valor de la heurística
    vector<Edge> neighbors;  // aristas que salen de este nodo

    Node(int id = -1, int x = 0, int y = 0, int h = 0)
        : id(id), x(x), y(y), heuristic(h) {
    }
};

class Graph {
@@ -31,7 +38,7 @@ class Graph {
        int id = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                nodes.emplace_back(id++, x, y, 0);
            }
        }

@@ -49,7 +56,7 @@ class Graph {
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int to = index(nx, ny);
                            nodes[from].neighbors.emplace_back(to, 1);
                        }
                    }
                }
@@ -58,17 +65,24 @@ class Graph {
    }

    // Heurística tipo Chebyshev (movimientos en 8 direcciones con costo 1)
    static int chebyshevDistance(const Node& a, const Node& b) {
        return max(abs(a.x - b.x), abs(a.y - b.y));
    }

    // Asigna heurística a todos los nodos, dada la meta
    void setHeuristics(int goalId) {
        if (goalId < 0 || goalId >= nodes.size()) return;

        const Node& goalNode = nodes[goalId];
        for (auto& n : nodes) {
            n.heuristic = chebyshevDistance(n, goalNode);
        }
    }

    // Método para verificar si un nodo existe
    bool nodeExists(int nodeId) const {
        return nodeId >= 0 && nodeId < nodes.size();
    }
};

#endif