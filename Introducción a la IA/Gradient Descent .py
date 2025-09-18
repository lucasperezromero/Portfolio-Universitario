import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def Gradient_Descent(f,derivative , x_o):
    max_iter = 1000 #Numero maximo de iteraciones 
    n_t = 0.4 #Learning factor
    tol = 1e-12 #Tolerancia del error
    x_i = [x_o]
    if derivative(x_o) == 0:  
        print(f"La derivada en {x_i[-1]} es 0 y por tanto no puede se puede aplicar el método de Newton ")
        
    for i in range(max_iter):
        x_n = x_i[-1] - n_t * derivative(x_i[-1])
        
        if abs(x_i[-1] - x_n) < tol:
                break
        x_i.append(x_n)
    return x_i
#Función 1D
x = sp.symbols('x')
y = sp.symbols('y')
f_symb = (x - 3)**2 + (y +2)**2 + x*y
f = sp.lambdify(x, f_symb, 'numpy') 
derivative_x = sp.lambdify(x, sp.diff(f_symb, x), 'numpy')
derivative_y = sp.lambdify(y, sp.diff(f_symb, y), 'numpy')


x_o = 1
y_o = 0



minimum_x = Gradient_Descent(f,derivative_x, x_o)
minimum_y = Gradient_Descent(f,derivative_y,y_o)
    
print(f"El valor que hace mínima la función en x {f_symb} es : {minimum_x[-1]} en {len(minimum_x)} iteraciones")
print(f"El valor que hace mínima la función en y {f_symb} es : {minimum_y[-1]} en {len(minimum_y)} iteraciones")



