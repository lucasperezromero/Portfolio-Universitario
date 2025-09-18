import random
import math

# Definir las propiedades del sistema
num_particulas = 200  # Número de partículas
tamano_espacio = 10 # Tamaño del área cuadrada (10x10)
radio_particula = 0.5  # Radio de las partículas (para la colisión)
num_simulaciones = 100  # Número de simulaciones a realizar
posiciones = []

def simular_gas(num_particulas, tamano_espacio, radio_particula, num_simulaciones):
    colisiones_totales = 0

    for _ in range(num_simulaciones):
        # Crear posiciones aleatorias para las partículas dentro del espacio
        for _ in range(num_particulas):
            posiciones.append([random.uniform(radio_particula, tamano_espacio - radio_particula),
                            random.uniform(radio_particula, tamano_espacio - radio_particula),
                            random.uniform(radio_particula, tamano_espacio - radio_particula)])
                     
   
    # Contar las colisiones
    colisiones = 0
    for i in range(num_particulas):
        for j in range(i+1, num_particulas):
            distancia = math.sqrt((posiciones[i][0] - posiciones[j][0])**2 +
                                  (posiciones[i][1] - posiciones[j][1])**2 +
                                  (posiciones[i][2] - posiciones[j][2])**2)
            # Si la distancia entre dos partículas es menor que la suma de sus radios, ocurre una colisión
            if distancia < 2 * radio_particula:
                colisiones += 1
       
        # Sumar las colisiones en esta simulación
        colisiones_totales += colisiones
   
    # Calcular el número promedio de colisiones
    colisiones_promedio = colisiones_totales / num_simulaciones
    return colisiones_promedio

# Simular el gas y obtener el número promedio de colisiones
colisiones_promedio = simular_gas(num_particulas, tamano_espacio, radio_particula, num_simulaciones)

print(f"El número promedio de colisiones entre partículas en una simulación es: {colisiones_promedio:.2f}")


