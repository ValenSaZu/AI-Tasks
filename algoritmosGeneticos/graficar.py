import pandas as pd
import matplotlib.pyplot as plt

# 1. Leer el CSV generado por main.cpp
data = pd.read_csv("resultados.csv")

# 2. Crear la figura
plt.figure(figsize=(10,6))

# Curvas
plt.plot(data["Generacion"], data["Promedio"], label="Promedio", color="blue", linewidth=2)
plt.plot(data["Generacion"], data["Mejor"], label="Mejor", color="red", linewidth=2)

# 3. Dar formato
plt.title("Evolución del Algoritmo Genético", fontsize=14)
plt.xlabel("Generación", fontsize=12)
plt.ylabel("Aptitud", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# 4. Guardar y mostrar (no se cerrará hasta que cierres la ventana manualmente)
plt.savefig("grafica_resultados.png", dpi=300)
plt.show(block=True)
