import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def newton_method(f,derivative, x_o): #Function that calculates the Newton Method
    
    umbral = 1e-8
    # We create a list for aproximations
    lista_aprox = [x_o]

    # List for errors
    errors = []

    for x_i in lista_aprox:
        #calculation of x_i+1 in y
        y = x_i - f.subs(z,x_i).evalf()/derivative.subs(z,x_i).evalf()
        #añadimos el valor a la lista de aproximaciones
        lista_aprox.append(y)
        #añadimos el error a la lista de errores
        errors.append(abs(y-x_i))
        #comprovamos si tenemos la precision suficiente
        if umbral > errors[-1]:
            break
        #comprovamos a partir de la decima iteracion si la tendencia de los errores esta creciendo, en caso afirmativo detenemos el bucle
        if len(lista_aprox) > 10:
            if errors[-1] > errors[-2]:
                break
    #si se ha alcanzado la precision deseada devolvemos la ultima aproximación
    if umbral > errors[-1]:
        return lista_aprox[-1]
    #si no se ha alcanzado la precision porque el error divergia entonces avisamos por pantalla
    else:
        print(f"\nLa semilla {x_o} no converge a una raiz de la funcion {f}.")
        return None

def Root_finder(f, derivative,seeds): #Function that finds the intersection points
    raices=[]
    #recorremos la lista de semillas para buscar todas las raices de la función
    for s in seeds:
        #añadimos a la lista de raices el return de la funcion del metodo de newton
        raices.append(newton_method(f, derivative, s))
    return raices

z = sp.symbols('z') #function variable
a= 3e-10 #m
V_0 = 4.806e-19 #J
hbar = 1.054e-34 #J
m_eff = 0.8*9.109e-31 #kg
z_o = np.sqrt((2 * m_eff * V_0 * a**2) / hbar**2) #Parameter

def Even_Function(z_o, z ):
    arg = (z_o/z)**2 - 1 #Squareroot argument
    f_1 = sp.cot(z) 
    f_2 = sp.sqrt(arg)
    
    f = f_1+ f_2 #Transcendental Even Function
    
    derivative = sp.diff(f, z)
    x_o = 1
    seeds = np.arange(x_o,z_o,np.pi)
    roots = Root_finder(f,derivative, seeds) #Calculates the roots for energy levels 
    
    #Function Plot in order to see the roots
    f_plot = sp.lambdify(z, f, 'numpy') #Makes the function numerical
    z_vals = np.linspace(0.2, float(z_o)-0.1, 1000)
    f_vals = []
    for val in z_vals:
        try:
            arg = (z_o/val)**2 - 1
            f_vals.append(f_plot(val)) # Evaluate and store the function value at each z
        except:
            f_vals.append(np.nan) #If there is an error it saves it as an empty slot 

    plt.figure()
    plt.plot(z_vals, f_vals, label='cotg(z) + sqrt((z_o/z)^2 - 1)')
    plt.axhline(0, color='r', linestyle='--')
    plt.ylim(-10, 10)
    plt.xlabel("z")
    plt.ylabel("f(z)")
    plt.title("Transcendental Equation (Even)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return roots


def Odd_Function(z_o, z ):
    arg = (z_o/z)**2 - 1 # Squareroot argument
    f_1 = sp.tan(z) 
    f_2 = sp.sqrt(arg)
    
    f = f_1 - f_2 #Transcendental Odd Function
    derivative = sp.diff(f, z)
    x_o = 1
    seeds = np.arange(x_o,z_o,np.pi)
    roots = Root_finder(f,derivative, seeds) #Calculates the roots for energy levels 
    
    #Function Plot in order to see the roots
    f_plot = sp.lambdify(z, f, 'numpy') #Makes the function f numerical 
    z_vals = np.linspace(0.2, float(z_o) - 0.1, 1000) #Generates values of z for the plot
    f_vals = []
    for val in z_vals:
        try:
            arg = (z_o/val)**2 - 1
            f_vals.append(f_plot(val)) # Evaluate and store the function value at each z
        except:
            f_vals.append(np.nan) #If there is an error it saves it as an empty slot 

    plt.figure()
    plt.plot(z_vals, f_vals, label='tg(z) + sqrt((z_o/z)^2 - 1)')
    plt.axhline(0, color='r', linestyle='--')
    plt.ylim(-10, 10)
    plt.xlabel("z")
    plt.ylabel("f(z)")
    plt.title("Transcendental Equation (Odd)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    return roots


X_e = Even_Function(z_o, z) #Even Function roots
X_o = Odd_Function(z_o, z) #Odd Function roots
print(f"The roots for the odd function are {X_o[0]}")
print(f"The roots for the even function are {X_e[0]}")

E_1 = ( hbar*X_o[0]/a)**2 *(1/2*m_eff) #First energy level 
E_2 = ( hbar*X_e[0]/a)**2 *(1/2*m_eff) #Second energy level

print(f"The two energy levels are {E_1} J and {E_2} J")

energies= [E_1/1.602e-19 , E_2/1.602e-19] # Makes an array of energies in eV
plt.figure(figsize=(4,6))
for i, e in enumerate(energies):
    plt.hlines(e, 0, 1, colors='blue', linewidth=3) #Makes the horizontal lines of the energy levels
    plt.text(1.05, e, f'n={i+1}', verticalalignment='center')
    
plt.ylabel("Energy (eV)")
plt.title("Energy spectrum of finite quantum well\nwith energy gap")
plt.xticks([])
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.show()