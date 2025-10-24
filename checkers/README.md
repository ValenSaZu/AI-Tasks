# Checkers MinMax en OpenGL

**Laboratorio 02 - Entrega: 05 Agosto 2025**  
Implementación del juego de damas simple (sin reinas) con el algoritmo MinMax.

## Descripción
Este proyecto implementa **el juego de damas** donde:
1. El usuario juega contra la computadora.
2. Al inicio, antes de abrir la pantalla del jeugo, se necesita ingresar un nivel de dificultad que es la altura de analisis del arbol de decisiones con el que jugará la computadora.
3. Se implementa un árbol con el algoritmo MinMax recursivo donde cada nodo es el tablero después de alguna decisión (este nodo contiene un vector de nodos, la jugada y la jugada que lo llevó a allí), la computadora supone que el usuario elegirá la peor jugada para él (Min) y toma la mejor jugada para él (Max), la heurística usada es el número de fichas de la computadora - el número de fichas del usuario, negativo en favor del usuario y postivo en favor de la computadora, el vector de nodos hijos se evalua comidas y movimientos simples de cada ficha, solo se ingresan aquellos movimientos válidos.
4. Visualización de:
   - **Fichas**: Fichas del usuario (color negras) y fichas del computador (color rojas).
   - **Tablero**: Tablero (color beige y marrón) como matriz, las fichas se posicionan en el color oscuro.
   - **Selección de ficha a jugar(contorno verde)**
   - **Texto de estado**: Turno, si hay ganador, dificultad en la que se setteo el juego e instrucciones.
