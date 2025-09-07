# Graph Coloring con OpenGL

**Laboratorio 02 - Constraint Satisfaction Problems (CSP)**  
**Fecha de Entrega: 12 de Agosto de 2025**

Una implementación interactiva de algoritmos de coloreo de grafos (Graph Coloring) como un Problema de Satisfacción de Restricciones (CSP), con visualización gráfica usando OpenGL.

## Descripción

Este proyecto genera de forma aleatoria un grafo no dirigido y permite resolver el problema de colorearlo utilizando diferentes algoritmos de búsqueda. La aplicación visualiza el proceso en una ventana OpenGL, mostrando los nodos, las aristas y la asignación de colores, e incluye una alerta visual cuando ocurre un backtracking.

## Características

1. **Generación de Grafos:** Crea un grafo con un número inicial de nodos y un número de colores definidos por el usuario. Cada nodo establece entre 1 y 4 aristas aleatorias con otros nodos.
2. **Algoritmos de Solución:** Implementa diferentes estrategias para resolver el CSP:
    -   **Backtracking (BT):** Búsqueda por fuerza bruta con vuelta atrás.
    -   **Backtracking con Forward Checking (BT-FC):** Mejora el BT eliminando valores inconsistentes de los dominios de las variables vecinas no asignadas.
    -   **Backtracking con Propagación de Restricciones (e.g., MAC):** Una versión más robusta que mantiene una consistencia stronger (como arco-consistencia) durante la búsqueda.
3. **Visualización Interactiva:**
    -   Dibuja el grafo con nodos y aristas.
    -   Muestra el coloreo resultante o el proceso paso a paso.
    -   Indica visualmente (por ejemplo, cambiando el color de fondo o de un nodo) cuándo ocurre un backtracking.
    -   Interfaz para seleccionar el algoritmo de búsqueda (mediante botones en la ventana o entradas por teclado/consola).