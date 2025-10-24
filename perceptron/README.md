# Perceptron Simple para CIFAR-10

## Requisitos del Proyecto

### Arquitectura del Modelo
- **Perceptron Simple**: Una sola capa sin capas ocultas
- **10 Neuronas**: Una por cada clase de CIFAR-10
- **Entrada**: Imágenes RGB 32x32 (3,072 características), las imagenes se almacenan en formato plano con 3073 bytes
      - 1 byte para la etiqueta (valor entre 0-9)
      - 3072 bytes para los pixeles de la imagen: 32×32×3 = 3072. Los primeros 1024 bytes son el canal rojo (R) ordenado fila-por-fila, luego los 1024 del canal verde (G), luego los 1024 del canal azul (B). 
      - Por tanto cada “fila” es de 3073 bytes.
- **Salida**: Probabilidades para 10 clases usando función softmax

### Implementación Técnica
- **Lenguaje**: C++ 
- **Aceleración GPU**: Implementación con CUDA
- **Operaciones**: Cálculos matriciales optimizados para GPU

### Hiperparámetros de Entrenamiento
- **batch_size**: 256 muestras por lote
- **Épocas**: Múltiples iteraciones sobre el dataset
- **Muestreo**: 256 muestras aleatorias por época
- **Learning Rate**: A definir según experimentación

### Métricas a Monitorear
- **Error**: Tasa de clasificación incorrecta
- **Pérdida (Loss)**: Distancia euclidiana entre predicción y objetivo
- **Accuracy**: Porcentaje de clasificación correcta

### Procesamiento de Datos
- **Dataset**: CIFAR-10 - imágenes color RGB 32x32
- **Estructura**: 3 matrices (canales R, G, B) por imagen
- **Clases**: 10 categorías diferentes

## Estructura del Proyecto

```
Entrada: 32x32x3 = 3,072 características
    ↓
Capa Linear: 3,072 → 10 neuronas
    ↓
Función Softmax
    ↓
Salida: 10 probabilidades (una por clase)
```

## Clases CIFAR-10
```
0: airplane    1: automobile  2: bird
3: cat        4: deer        5: dog  
6: frog       7: horse       8: ship
9: truck
```

## Componentes Principales
- Forward pass con cálculo de activaciones
- Backward pass para cálculo de gradientes
- Actualización de pesos mediante descenso de gradiente
- Procesamiento por lotes con muestreo aleatorio
- Métricas de evaluación en tiempo real

**Nota**: Implementación optimizada para GPU usando CUDA para operaciones matriciales.

## Referencias
- Descargar Nvidia CUDA Toolkit para Windows: https://www.youtube.com/watch?v=4wPUtUtSp-o
   - pagina oficial para descargar: https://developer.nvidia.com/cuda-downloads

- Dataset CIFAR-10 binario para C/C++: https://www.cs.toronto.edu/~kriz/cifar.html
   - clase cifar10_reader basada en: https://github.com/wichtounet/cifar-10/blob/master/include/cifar/cifar10_reader.hpp

- Softmax: https://www.youtube.com/watch?v=ma-F0RsMAjQ


