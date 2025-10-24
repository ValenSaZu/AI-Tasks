# Pathfinding en OpenGL

**Laboratorio 01 - Entrega: 29 Agosto 2025**  
Implementación de búsquedas ciegas y heurísticas en un grafo generado dinámicamente.

## Descripción
Este proyecto simula la generación de un **grafo dinámico** donde:
1. Los nodos se conectan hacia sus vecinos (arriba, abajo, diagonales) según su posición.
2. Un porcentaje de nodos se elimina aleatoriamente, es necesario ingresar este porcentaje para que se genere el grafo.
3. Se implementan 4 algoritmos de búsqueda para encontrar un camino entre un nodo inicial y uno final:
   - **Búsquedas ciegas**: BFS y DFS
   - **Búsquedas heurísticas**: Hill Climbing, Best-First y A*
4. Visualización de:
   - **Exploración**: Rutas evaluadas por el algoritmo (color azul).
   - **Camino final**: Ruta óptima encontrada (color rojo).
   - **Selección de nodo inicial(verde)/final(rojo) (clic).**
   - **Botones para cambiar algoritmos.**
