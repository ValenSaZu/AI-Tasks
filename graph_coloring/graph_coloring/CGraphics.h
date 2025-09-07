#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <ctime>

using namespace std;

struct Edge {
    int destination; // índice del nodo de destino
    int weight;      // costo de la arista

    Edge(int dest, int w = 1) : destination(dest), weight(w) {}
};

struct Node {
    int id;                  // identificador del nodo
    int heuristic;           // valor de la heurística
    vector<Edge> neighbors;  // aristas que salen de este nodo

    Node(int id = -1, int x = 0, int y = 0, int h = 0)
        : id(id), heuristic(h) {
    }
};

class Graph {
public:
    vector<Node> nodes; // lista de nodos

	// Genera el grafo con las aristas aleatorias
	//TODO: implementar la generación de aristas para que guarde en su heuristica a cuantos nodos está conectado
    void buildGraph(int num) {
        nodes.clear();
        nodes.reserve(num);

        // Crear nodos
        int id = 0;
        for (int x = 0; x < num; x++) {
            nodes.emplace_back(id++, 0);
        }
    }
};

#endif#pragma once
