import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Leer el CSV generado por main.cpp
data = pd.read_csv("resultados.csv")

# 2. Crear la figura con mejor tamaño
plt.figure(figsize=(12, 8))

# 3. Configurar el gráfico para enfatizar la tendencia descendente
plt.plot(data["Generacion"], data["Mejor"], 
         label="Mejor distancia", color="red", marker='o', 
         markersize=4, linewidth=2.5)
plt.plot(data["Generacion"], data["Promedio"], 
         label="Distancia promedio", color="blue", linestyle='--', 
         marker='x', markersize=4, linewidth=2)

# 4. Configurar ejes para mejor visualización vertical
plt.xlabel("Generación", fontsize=12, fontweight='bold')
plt.ylabel("Valor de Aptitud (Minimización)", fontsize=12, fontweight='bold')
plt.title("Evolución del Algoritmo Genético - Minimización de f(x,y) = x² - y² + 2xy", 
          fontsize=14, fontweight='bold', pad=20)

# 5. Invertir el eje Y para mostrar mejora hacia abajo más claramente
plt.gca().invert_yaxis()

# 6. Mejorar la visualización
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)

# 7. Configurar límites para mejor visualización
y_min = min(data["Mejor"].min(), data["Promedio"].min())
y_max = max(data["Mejor"].max(), data["Promedio"].max())
y_range = y_max - y_min
plt.ylim(y_max + y_range*0.1, y_min - y_range*0.1)  # Invertido para mostrar mejora hacia abajo

# 8. Agregar anotaciones para mostrar la mejora
inicio_mejor = data["Mejor"].iloc[0]
final_mejor = data["Mejor"].iloc[-1]
mejora = inicio_mejor - final_mejor

plt.annotate(f'Inicio: {inicio_mejor:.2f}', 
             xy=(0, inicio_mejor), xytext=(10, inicio_mejor + y_range*0.15),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
             fontsize=10, color='red')

plt.annotate(f'Final: {final_mejor:.2f}\nMejora: {mejora:.2f}', 
             xy=(data["Generacion"].iloc[-1], final_mejor), 
             xytext=(data["Generacion"].iloc[-1] - 15, final_mejor - y_range*0.15),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
             fontsize=10, color='red')

# 9. Ajustar diseño y guardar
plt.tight_layout()
plt.savefig("grafica_resultados_vertical.png", dpi=300, bbox_inches='tight')
plt.show()

# 10. Estadísticas adicionales
print(f"Mejora total: {mejora:.2f}")
print(f"Mejora porcentual: {(mejora/abs(inicio_mejor))*100:.2f}%")
print(f"Mejor aptitud encontrada: {final_mejor:.2f}")
print(f"Generación con mejor resultado: {data.loc[data['Mejor'].idxmin(), 'Generacion']}")