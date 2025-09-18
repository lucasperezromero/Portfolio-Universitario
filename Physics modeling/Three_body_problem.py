import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp

def Ecuacion_Euler (M_1 , M_2, M_3 ):
    l = sp.symbols('l')
    eq = (M_1 + M_2)*l**5 + (3*M_1 + 2*M_2)*l**4 + (3*M_1 + M_2)*l**3 - (3*M_3 + M_2)*l**2 - (3*M_3 + 2*M_2)*l - (M_3 + M_2)
    soluciones = sp.solve(eq, l)
    sol_numericas = [sol.evalf() for sol in soluciones if sol.is_real and sol > 0]
    return float(sol_numericas[0])

def Ecuaciones_Movimiento (t, y):
    x_1, y_1, x_2, y_2, x_3, y_3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    
    eps = 1e-4
    r12 = np.array([x_2 - x_1 ,y_2 - y_1])
    r13 = np.array([x_3 - x_1 ,y_3 - y_1])
    r23 = np.array([x_3 - x_2 ,y_3 - y_2])
    
    r12_mag = max(np.linalg.norm(r12), eps)
    r13_mag = max(np.linalg.norm(r13), eps)
    r23_mag = max(np.linalg.norm(r23), eps)
    
    # Aceleraciones
    a1 = G * (M_2 * r12 / r12_mag**3 + M_3 * r13 / r13_mag**3)
    a2 = G * (M_3 * r23 / r23_mag**3 - M_1 * r12 / r12_mag**3)
    a3 = G * (-M_1 * r13 / r13_mag**3 - M_2 * r23 / r23_mag**3)
   
    return [vx1, vy1, vx2, vy2, vx3, vy3, a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]]


def Puntos_Lagrange (y,M_1, M_2):
    x_1, y_1, x_2, y_2 = y
   
    #Coordenadas y CM
    x_12 = x_2 - x_1
    y_12 = y_2 - y_1
    x_cm = (x_1* M_1 + x_2*M_2 )/(M_1 + M_2) #Nuevo CM en x
    y_cm = (y_1*M_1 + y_2*M_2)/(M_1 + M_2) #Nuevo CM en y
    
    #Definimos incognitas
    x = sp.symbols('x')
    y = sp.symbols('y')
    
    #Ecuaciones L1 para las coordenadas x e y
    eq_x = M_1/(x_12 - x)**2 - M_2/x**2 - M_1*(x_12 - x - x_cm)/x_12**3 
    eq_y =  M_1/(y_12 - y)**2 - M_2/y**2 - M_1*(y_12 - y - y_cm)/y_12**3 
    sol_x = sp.solve(eq_x, x) #Obtiene la solución en formato sympy (simbolico)
    sol_y = sp.solve(eq_y,y)
    sol_x_numericas = [sol.evalf() for sol in sol_x if sol.is_real and sol > 0] #Pasa la solución a formato numerico
    sol_y_numericas = [sol.evalf() for sol in sol_y if sol.is_real and sol > 0]
    L_1 = [sol_x_numericas[0],sol_y_numericas[0]]
    
    #Redefino variables para las ecuaciones de L2
    eq_x = M_1/(x_12 + x)**2 + M_2/x**2 - M_1*(x_12 + x - x_cm)/x_12**3 
    eq_y = M_1/(y_12 + y)**2 + M_2/y**2 - M_1*(y_12 + y - y_cm)/y_12**3 
    sol_x = sp.solve(eq_x, x)
    sol_y = sp.solve(eq_y,y)
    sol_x_numericas = [sol.evalf() for sol in sol_x if sol.is_real and sol > 0] #Pasa la solución a formato numerico
    sol_y_numericas = [sol.evalf() for sol in sol_y if sol.is_real and sol > 0]
    L_2 = [sol_x_numericas[0],sol_y_numericas[0]]
    
    #Redefino variables para las ecuaciones de L3
    eq_x = M_1/(x - x_12)**2 + M_2/x**2 - M_1*(x - x_12  + x_cm)/x_12**3 
    eq_y = M_1/(y - y_12)**2 + M_2/y**2 - M_1*(y - y_12 + y_cm)/y_12**3 
    sol_x = sp.solve(eq_x, x)
    sol_y = sp.solve(eq_y,y)
    sol_x_numericas = [sol.evalf() for sol in sol_x if sol.is_real and sol > 0] #Pasa la solución a formato numerico
    sol_y_numericas = [sol.evalf() for sol in sol_y if sol.is_real and sol > 0]
    L_3 = [sol_x_numericas[0],sol_y_numericas[0]]
    
    #Para L4 y L5 usaremos el teorema del seno, dado que forman triangulos equilateros
    theta = np.pi/3
    L_4 = [x_1 + (x_2 - x_1)*np.cos(theta) - (y_2 - y_1)*np.sin(theta), y_1 + (y_2 - y_1)*np.cos(theta) + (x_2 - x_1)*np.sin(theta)]
    L_5 = [x_1 + (x_2 - x_1)*np.cos(-theta) - (y_2 - y_1)*np.sin(-theta), y_1 + (y_2 - y_1)*np.cos(-theta) + (x_2 - x_1)*np.sin(-theta)]
       
    print(f"El punto L1 está en {L_1}")
    print(f"El punto L2 está en {L_2}")
    print(f"El punto L3 está en {L_3}")
    print(f"El punto L4 está en {L_4}")
    print(f"El punto L5 está en {L_5}")
    
   
    # Primero ploteamos los puntos
    plt.scatter(x_1, y_1, label="M_1")
    plt.scatter(x_2, y_2, label="M_2")
    plt.scatter(L_1[0], L_1[1], label="L1")
    plt.scatter(L_2[0], L_2[1], label="L2")
    plt.scatter(L_3[0], L_3[1], label="L3")
    plt.scatter(L_4[0], L_4[1], label="L4")
    plt.scatter(L_5[0], L_5[1], label="L5")
    
    # Línea entre M1 y M2
    plt.plot([x_1, x_2, L_1[0],L_2[0] ,L_3[0]], [y_1, y_2, L_1[1],L_2[1] ,L_3[1]], 'k--', label="Linea M1-L3")
    
    # Triángulo con L4
    plt.plot([x_1, L_4[0], x_2], [y_1, L_4[1], y_2], 'b-', alpha=0.6, label="Triángulo L4")
    
    # Triángulo con L5
    plt.plot([x_1, L_5[0], x_2], [y_1, L_5[1], y_2], 'r-', alpha=0.6, label="Triángulo L5")
    
    plt.legend()
    plt.xlabel('Posición X')
    plt.ylabel('Posición Y')
    plt.title('Distribución de puntos de Lagrange')
    plt.grid(True)
    plt.axis('equal')  # Para que los triángulos no salgan deformados
    plt.show()
        


#Constante de Gravitación
G_real = 6.67430e-11 #N·m^2/Kg^2
Factor_escala = 384400000/5 #Distancia Tierra-Luna /5
G =(G_real * 5.9722e24) / Factor_escala**2


#Masas terrestres
M_1 = 10
M_2 = 20
M_3 = 30

l = Ecuacion_Euler(M_1, M_2, M_3)
x_1 = 20
y_1 = x_1
x_2 = l*x_1
y_2 = x_2
x_3 = -(1 + l)*x_1
y_3 = x_3


#Velocidades iniciales
x_cm= (M_1*x_1+M_2*x_2+M_3*x_3)/(M_1 + M_2 + M_3) #Centro de masas en x
y_cm= (M_1*y_1+M_2*y_2+M_3*y_3)/(M_1 + M_2 + M_3) #Centro de masas en y
r_total = np.sqrt((x_3 - x_1)**2 + (y_3 - y_1)**2) 

w = np.sqrt(G*(M_1 + M_2 + M_3)/r_total**3) #Velocidad angular

#Velocidades Lineales
vx1 = w*(y_1 - y_cm) 
vx2 = w*(y_2 - y_cm)
vx3 = w*(y_3 - y_cm)
vy1 = w*(x_1 - x_cm)
vy2 = w*(x_2 - x_cm)
vy3 = w*(x_3 - x_cm)


#Resolución del sistema
y_0 = [x_1, y_1, x_2, y_2, x_3, y_3, vx1, vy1, vx2, vy2, vx3, vy3]
t_span = [0,1e3]
sol = solve_ivp(Ecuaciones_Movimiento, t_span, y_0, method='DOP853', rtol = 1e-10)
y_0 = [x_1, y_1, x_2, y_2]
Puntos_Lagrange(y_0, M_1, M_2)

# Graficar posiciones
plt.plot(sol.t, sol.y[0], label="M_1")
plt.plot(sol.t, sol.y[2], label="M_2")
plt.plot(sol.t, sol.y[4], label="M_3")
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Posición X')
plt.title('Posiciones en X')
plt.show()

plt.plot(sol.t, sol.y[1], label="M_1")
plt.plot(sol.t, sol.y[3], label="M_2")
plt.plot(sol.t, sol.y[5], label="M_3")
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Posición Y')
plt.title('Posiciones en Y')
plt.show()

