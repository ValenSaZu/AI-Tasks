#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <random>
#include <iostream>

using namespace std;

//A tomar en cuenta
// - n_aristas_posibles = n*(n-1)/2 -> n es el numero de nodos
// - Densidad = n_aristas / n_aristas_posibles
// Entonces, si el profesor nos dice que en un grafo de 8 nodos, no debe haber mas de 4 aristas por nodo:
//      - n_aristas_posibles = 8*(8-1)/2 = 28
//      - n_aristas = 8*4/2 = 16 (se divide entre 2 porque cada arista conecta dos nodos)
//      - Densidad = 16/28 = 0.57

struct Edge {
    int destination; // índice del nodo de destino
    Edge(int dest) : destination(dest) {}
};

struct Node {
    int id;                  // identificador del nodo
    int heuristic;           // valor de la heurística (grado del nodo)
    vector<Edge> neighbors;  // aristas que salen de este nodo
    int color = NULL;           // color asignado al nodo
    vector<int>Colors;       // colores que se pueden usar en el nodo

    float x, y; // Agregar coordenadas para visualización

    Node(int id = -1, int h = 0)
        : id(id), heuristic(h) {
    }
};

class Graph {
private:
    mt19937 gen; //generador de numeros aleatorios, mucho mejor que rand()

public:
    vector<Node> nodes; // lista de nodos

    Graph(int num, double density = 0.57) {
        // Inicializar generador de números aleatorios
        random_device rd;
        gen.seed(rd());
        buildGraph(num, density);
    }

    // Genera el grafo con las aristas aleatorias
    void buildGraph(int num, double density) {
        nodes.clear();
        nodes.reserve(num);

        // Crear nodos
        for (int i = 0; i < num; i++) {
            nodes.emplace_back(i, 0);
        }

        // Generar aristas aleatorias con densidad controlada
        generateRandomEdges(density);

        // Calcular heurística (grado de cada nodo)
        calculateHeuristics();
    }

    // Genera aristas aleatorias evitando grafo completo
    void generateRandomEdges(double maxDensity) {
        int n = nodes.size();
        if (n <= 1) return;

        // Ruleta para probabilidades (decidir si agregar arista), esto nos permitir[a no tener siempre el mismo numero de aristas
        uniform_real_distribution<double> dist(0.0, 1.0);
        // Dado para números enteros (elegir nodos)
        uniform_int_distribution<int> nodeDist(0, n - 1);

        // Máximo número de aristas para evitar grafo demasiado denso, podria dar un numero no entero, por eso se castea a int
        int maxEdges = static_cast<int>((n * (n - 1) / 2) * maxDensity);
        // Evita que pueda salir un numero muy bajo como 0, mínimo n-1 aristas para que el grafo sea conexo
        int minEdges = n - 1;
        int currentEdges = 0;

        uniform_int_distribution<int> n_edges(minEdges, maxEdges);

        maxEdges = n_edges(gen); // ahora si, numero aleatorio de aristas

        // Primero asegurar que el grafo sea conexo, de esta manera se evita que queden nodos aislados
        for (int i = 1; i < n; i++) {
            int j = nodeDist(gen) % i; // Conectar con un nodo anterior aleatorio
            if (j != i) {// no se conecta consigo mismo
                addEdge(i, j);
                currentEdges++;
            }
        }

        // Agregar aristas adicionales hasta alcanzar las aristas
        while (currentEdges < maxEdges) {
            int u = nodeDist(gen);
            int v = nodeDist(gen);
            if (u != v && !edgeExists(u, v) && dist(gen) < 0.7) {
                addEdge(u, v);
                currentEdges++;
            }
        }
    }

    // Verifica si ya existe una arista entre dos nodos
    bool edgeExists(int u, int v) {
        for (const auto& edge : nodes[u].neighbors) {
            if (edge.destination == v) return true;
        }
        for (const auto& edge : nodes[v].neighbors) {
            if (edge.destination == u) return true;
        }
        return false;
    }

    // Añade una arista bidireccional
    void addEdge(int u, int v) {
        nodes[u].neighbors.emplace_back(v);
        nodes[v].neighbors.emplace_back(u);
    }

    // Calcula la heurística (grado de cada nodo)
    void calculateHeuristics() {
        for (auto& node : nodes) {
            node.heuristic = node.neighbors.size();
        }
    }

    // Ordenar nodos por heurística (grado) de forma descendente
    void sortByHeuristicDesc() {
        sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
            return a.heuristic > b.heuristic;
            });
    }

    // Ordenar nodos por número de colores disponibles, del menor al mayor
    void sortByAvailableColors(int begin) {
        sort(nodes.begin() + begin, nodes.end(), [](const Node& a, const Node& b) {
            return a.Colors.size() < b.Colors.size();
            });
    }

    // Ordenar nodos por ID
    void sortById() {
        sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
            return a.id < b.id;
            });
    }

    // Añade los colores posibles a cada nodo del 0 a nColors-1
    void addColors(int nColors) {
        for (auto& node : nodes) {
            node.Colors.clear();
            for (int c = 0; c < nColors; c++) {
                node.Colors.push_back(c);
            }
        }
    }

    // Limpia los colores asignados y los dominios
    void cleanAll() {
        for (auto& node : nodes) {
            node.color = NULL;
            node.Colors.clear();
        }
    }

    void printColoring() {
        cout << "Coloreo del grafo:" << endl;
        for (const auto& node : nodes) {
            cout << "Nodo " << node.id << " -> Color " << node.color << endl;
        }
    }
};

#endif