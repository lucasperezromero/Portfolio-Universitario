import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def newton_method(f,derivative, x_o): #Función que realiza el método de Newton
    max_iter = 1000 #Numero maximo de iteraciones 
    tol = 1e-12 #Tolerancia del error
    x_i = [x_o]
    if derivative(x_o) == 0:  
        return print(f"La derivada en {x_i[-1]} es 0 y por tanto no puede se puede aplicar el método de Newton ")
        
    for i in range(max_iter):
        x_n = x_i[-1] - f(x_i[-1])/derivative(x_i[-1])
        
        if abs(x_i[-1] - x_n) < tol:
                break
        x_i.append(x_n)
    return x_i
    

#Función 1D
x = sp.symbols('x')
f_symb = x**2 - 4*x + 4
f = sp.lambdify(x, f_symb, 'numpy') 
derivative = sp.lambdify(x, sp.diff(f_symb, x), 'numpy')
x_o = float(input(f"Escoge un valor para iniciar el método de Newton con la función {f_symb}: "))
raiz = newton_method(f,derivative,x_o)

print(f"Raíz aproximada de la función {f_symb}: {raiz[-1]} en {len(raiz)} iteraciones")

