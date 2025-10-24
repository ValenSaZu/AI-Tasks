import pandas as pd
import matplotlib.pyplot as plt

# 1. Leer el CSV generado por algoritmo genético
data = pd.read_csv("resultados.csv")

# 2. Crear figura del mismo tamaño que tu imagen
plt.figure(figsize=(10, 6))

# 3. Graficar exactamente como tu imagen
plt.plot(data["Generacion"], data["Promedio"], 
         label="Promedio (avg)", color="#1f77b4", marker='o', 
         markersize=2, linewidth=1.2)
plt.plot(data["Generacion"], data["Mejor"], 
         label="Mejor (best)", color="#ff7f0e", marker='s', 
         markersize=2, linewidth=1.2)

# 4. Configurar exactamente como tu imagen
plt.xlabel("Generación")
plt.ylabel("Valor de Función Fitness (Minimización)")
plt.title("Algoritmo Genético: Evolución de Aptitud")

# 5. Leyenda en la esquina superior derecha
plt.legend(loc='upper right', frameon=True, fontsize=9)

# 6. Grid sutil
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 7. Configurar ejes para mostrar la caída característica
plt.xlim(0, len(data)-1)

# Si los valores son muy negativos, ajustar el rango Y
y_min = min(data["Mejor"].min(), data["Promedio"].min())
y_max = max(data["Mejor"].max(), data["Promedio"].max())

# Para datos que van de 0 a valores muy negativos como -8000
if y_min < -1000:
    plt.ylim(y_min * 1.1, y_max * 1.1 if y_max > 0 else 0)
else:
    plt.ylim(y_min - abs(y_min)*0.1, y_max + abs(y_max)*0.1)

# 8. Ajustar formato
plt.tight_layout()
plt.savefig("grafica_ag_evolucion.png", dpi=300, bbox_inches='tight')
plt.show()

# 9. Mostrar estadísticas del proceso evolutivo
print("=== Estadísticas del Algoritmo Genético ===")
print(f"Aptitud inicial promedio: {data['Promedio'].iloc[0]:.2f}")
print(f"Aptitud final promedio: {data['Promedio'].iloc[-1]:.2f}")
print(f"Mejor aptitud inicial: {data['Mejor'].iloc[0]:.2f}")
print(f"Mejor aptitud final: {data['Mejor'].iloc[-1]:.2f}")
print(f"Mejora total (mejor): {data['Mejor'].iloc[0] - data['Mejor'].iloc[-1]:.2f}")
print(f"Generación de convergencia: {data.loc[data['Mejor'] == data['Mejor'].min(), 'Generacion'].iloc[0]}")