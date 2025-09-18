import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error       

#Parametros
k = 1
sigma = 0.9
N = 20000 #Numero de datos

#Generamos Datos
x = np.linspace(0 , 10, N)
noise = np.random.normal(0 , sigma , N)
F = -k * x + noise

#Separamos los datos
x_train , x_test , F_train , F_test = train_test_split(x , F , test_size = 0.5, random_state =42)

#Lista para almacenar errores
degrees = list(range(1,3))
E_in_values =[]
E_out_values = []

# Modelo Lineal
for d in degrees:
    
    poly = PolynomialFeatures(degree= d)
    x_train_poly = poly.fit_transform(x_train.reshape(-1 , 1))
    x_test_poly = poly.transform(x_test.reshape( -1 , 1))
    
    model = LinearRegression()
    model.fit(x_train_poly , F_train)
    
    #Predicciones
    F_train_pred = model.predict(x_train_poly)
    F_test_pred = model.predict(x_test_poly)
    
    #Calculo de errores
    E_in = mean_squared_error(F_train, F_train_pred) 
    E_out = mean_squared_error(F_test , F_test_pred)
    
    E_in_values.append(E_in)
    E_out_values.append(E_out)
    
    print(f"El error in-sample E_in para el polinomio de grado {d} es: {E_in}")
    print(f"El error out-sample E_out para el polinomio de grado {d} es: {E_out}")
    
    #Graficamos los datos
    plt.scatter(x_train , F_train , label= "Training Data" , color="blue", alpha=0.6)
    plt.scatter(x_test, F_test, label="Test data", color="red" , alpha=0.7)
    plt.plot(x, model.predict(poly.transform(x.reshape(-1 , 1))), label= "Modelo Ajustado", color="green" , alpha=0.7)            
    plt.xlabel("Desplazamiento (x)")
    plt.ylabel("Fuerza (F)")
    plt.title(f"Modelo ajustado en grado {d} ")
    plt.legend()
    plt.show()
    
plt.plot(degrees , E_in_values , label= "Evolución del Error de Entrenamiento (E_in)", marker ="o")
plt.plot(degrees , E_out_values , label= "Evolución del Error de Prueba (E_out)", marker = "s")
plt.xlabel("Grado del polinomio")
plt.ylabel("Errores del modelo")
plt.title("Comparación de errores conforme aumenta el grado del polinomio")
plt.legend()
plt.show()
    
    
