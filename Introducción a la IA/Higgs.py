import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def Gradient_Descent(f,derivative,φ_o,φ_th):
    max_iter = 1000
    tol = 1  # Tolerance
    n_t = 1e-5  # Learning Rate
    x_i = [φ_o]

    for i in range(max_iter):
        x_n = x_i[-1] - n_t*derivative(x_i[-1])
        x_i.append(x_n)
        tol_2 = φ_th - x_i[-1]

        if abs(x_n - x_i[-2]) < tol and tol_2 < 1:
            break

    return np.array(x_i)

def Gradient_Descent_2D(f,derivative_v_1,derivative_v_2,v_1_o,v_2_o, v_1_th,v_2_th):
    max_iter = 1000 #Iterations
    tol = 1  # Tolerance
    n_t = 1e-6  # Learning Rate
    x_i = np.array([[v_1_o,v_2_o]])

    for i in range(max_iter):
        x_1_n = x_i[-1][0] - n_t*derivative_v_1(x_i[-1][0],x_i[-1][1])
        x_2_n = x_i[-1][1] - n_t*derivative_v_2(x_i[-1][0],x_i[-1][1])
        x_i = np.vstack([x_i, [x_1_n, x_2_n]])
        tol_2 = v_1_th - x_i[-1][0]
        tol_3 = v_2_th - x_i[-1][1]

        if abs(x_1_n - x_i[-2][0]) < tol and abs(x_2_n - x_i[-2][1]) < tol and tol_2 < 1 and tol_3 < 1:
            break

    return x_i


def Gradient_Descent_Momentum(f, derivative, φ_o,φ_th):
    max_iter = 1000
    tol = 1  # Tolerance
    n_t = 1e-5  # Learning Rate
    gamma = 0.85  # Momentum parameter
    v = [0]
    x_i = [φ_o]

    for i in range(max_iter):
        v.append(gamma*v[-1] + n_t*derivative(x_i[-1]))
        x_n = x_i[-1] - v[-1]
        x_i.append(x_n)
        tol_2 = φ_th - x_i[-1]

        if abs(x_n - x_i[-2]) < tol and tol_2 < 1:
            break

    return np.array(x_i)
def Gradient_Descent_Momentum_2D(f,derivative_v_1,derivative_v_2,v_1_o,v_2_o, v_1_th,v_2_th):
    max_iter = 1000 #Iterations
    tol = 1  # Tolerance
    n_t = 1e-6  # Learning Rate
    gamma = 0.9  # Momentum parameter
    v = np.array([[0,0]])
    x_i =  np.array([[v_1_o, v_2_o]])

    for i in range(max_iter):
        v_1_n = gamma*v[-1][0] + n_t*derivative_v_1(x_i[-1][0],x_i[-1][1])
        v_2_n = gamma*v[-1][1] + n_t*derivative_v_2(x_i[-1][0],x_i[-1][1])
        v= np.vstack([v,[v_1_n,v_2_n]])
        
        x_1_n = x_i[-1][0] - v_1_n
        x_2_n = x_i[-1][1] - v_2_n
        
        x_i = np.vstack([x_i, [x_1_n, x_2_n]])
        tol_2 = v_1_th - x_i[-1][0]
        tol_3 = v_2_th - x_i[-1][1]

        if abs(x_1_n - x_i[-2][0]) < tol and abs(x_2_n - x_i[-2][1]) < tol and tol_2 < 1 and tol_3 < 1:
            break

    return x_i


def Nesterov_Accelerated_Gradient(f, derivative, φ_o,φ_th):
    max_iter = 1000
    tol = 1  # Tolerance
    n_t = 1e-5  # Learning Rate
    gamma = 0.85  # Momentum parameter
    v = [0]
    x_i = [φ_o]
    x_t = []

    for i in range(max_iter):
        x_t.append(x_i[-1] + gamma*v[-1])
        v.append(gamma*v[-1] + n_t*derivative(x_t[-1]))
        x_n = x_i[-1] - v[-1]
        x_i.append(x_n)
        tol_2 = φ_th - x_i[-1]

        if abs(x_n - x_i[-2]) < tol and tol_2 < 1:
            break

    return np.array(x_i)

def Nesterov_Accelerated_Gradient_2D(f,derivative_v_1,derivative_v_2,v_1_o,v_2_o, v_1_th,v_2_th):
    max_iter = 1000 #Iterations
    tol = 1  # Tolerance
    n_t = 1e-6  # Learning Rate
    gamma = 0.9  # Momentum parameter
    v = np.array([[0,0]])
    x_i =  np.array([[v_1_o, v_2_o]])
    x_t = []

    for i in range(max_iter):
        x_t_1_n = x_i[-1][0] + gamma*v[-1][0]
        x_t_2_n = x_i[-1][1] + gamma*v[-1][1]
        x_t.append([x_t_1_n,x_t_2_n])
        
        v_1_n = gamma*v[-1][0] + n_t*derivative_v_1(x_t[-1][0],x_t[-1][1])
        v_2_n = gamma*v[-1][1] + n_t*derivative_v_2(x_t[-1][0],x_t[-1][1])
        v= np.vstack([v,[v_1_n,v_2_n]])    
        
        x_1_n = x_i[-1][0] - v_1_n
        x_2_n = x_i[-1][1] - v_2_n
        x_i = np.vstack([x_i, [x_1_n, x_2_n]])      
        
        tol_2 = v_1_th - x_i[-1][0]
        tol_3 = v_2_th - x_i[-1][1]
        
        if abs(x_1_n - x_i[-2][0]) < tol and abs(x_2_n - x_i[-2][1]) < tol and tol_2 < 1 and tol_3 < 1:
            break

    return x_i


# Defino mi función
φ = sp.symbols('φ')
μ = 88.39  # GeV
λ = 0.129
φ_o = 100  # GeV
φ_th = 174.2  # GeV

f_symb = -μ**2*φ**2 + λ*φ**4
f = sp.lambdify(φ, f_symb, 'numpy')
derivative = sp.lambdify(φ, sp.diff(f_symb, φ), 'numpy')
# Theorical values 
v_th = np.sqrt(2)*φ_th
M_H_th = np.sqrt(2*λ*v_th**2)
# GD Results
GD_estimation = Gradient_Descent(f, derivative,φ_o,φ_th)
v_GD = np.sqrt(2)*GD_estimation
M_H_GD = np.sqrt(2*λ*v_GD**2)

# GDM Results 
GDM_estimation = Gradient_Descent_Momentum(f, derivative, φ_o,φ_th)
v_GDM = np.sqrt(2)*GDM_estimation
M_H_GDM = np.sqrt(2*λ*v_GDM**2)


# NAG Results
NAG_estimation = Nesterov_Accelerated_Gradient(f, derivative, φ_o,φ_th)
v_NAG = np.sqrt(2)*NAG_estimation
M_H_NAG = np.sqrt(2*λ*v_NAG**2)


print("Minimum estimation for each method:")
print("Gradient Descent: ",GD_estimation[-1])
print("Gradient Descent with Momentum: ",GDM_estimation[-1])
print("Nesterov Accelerated Gradient: ",NAG_estimation[-1])

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("Higgs boson mass estimation for each method:")
print(f"Gradient Descent: {M_H_GD[-1]} GeV")
print(f"Gradient Descent with Momentum is: {M_H_GDM[-1]} GeV")
print(f"Nesterov Accelerated Gradient is: {M_H_NAG[-1]} GeV")

plt.plot(range(0, len(M_H_GD)), M_H_GD,label='Mass estimation with GD', color='red')
plt.plot(range(0, len(M_H_GDM)), M_H_GDM,label='Mass estimation with GDM', color='blue')
plt.plot(range(0, len(M_H_NAG)), M_H_NAG,label='Mass estimation with NAG', color='green')

plt.axhline(y=M_H_th, color='black', linestyle='--',label='Theoretical Higgs Mass')
plt.xlabel("Iterations")
plt.ylabel("Mass estimation")
plt.title("Evolution of Mass estimation with each model iterations")
plt.legend()
plt.show()

#Part 2: Two Higgs Doublet Model

#Defining the potential and its variables 
v_1 = sp.symbols('v_1')
v_2 = sp.symbols('v_2')

#Potential Parameters
α_1 = 2.5e4 #GeV^2
α_2 = 2e4 #GeV^2
β_1 = 1
β_2 = 1 
β_3 = 0.5
#Potential and its derivatives
V_symb = -α_1*v_1**2 - α_2*v_2**2 + (β_1/2)*v_1**4 + (β_2/2)*v_2**4 + β_3*(v_1**2)*(v_2**2)
V = sp.lambdify((v_1,v_2), V_symb, 'numpy')
derivative_v_1 = sp.lambdify((v_1,v_2), sp.diff(V_symb, v_1), 'numpy') #Derivada respecto v_1
derivative_v_2 = sp.lambdify((v_1,v_2), sp.diff(V_symb, v_2), 'numpy') #Derivada respecto v_2
#Seed
v_o = 10 #GeV

#Theorical Higgs fields
v_1_th = 141.42  # GeV
v_2_th = 100 #GeV

#Theorical Masses
m_H_th = np.sqrt(β_1*v_1_th**2 + β_2*v_2_th**2 + abs(β_1*v_1_th**2 - β_2*v_2_th**2))
m_h_th = np.sqrt(β_1*v_1_th**2 + β_2*v_2_th**2 - abs(β_1*v_1_th**2 - β_2*v_2_th**2))
m_H_charged_th = np.sqrt(β_3*(v_1_th**2 + v_2_th**2))

#GD_2D Prediction
GD_estimation_2D = Gradient_Descent_2D(V,derivative_v_1,derivative_v_2,v_o,v_o,v_1_th,v_2_th)
#Separamos las funciones de onda v1 y v2 en sus vectores y gestionar mejor los datos
v_1_GD = GD_estimation_2D[:,0]
v_2_GD = GD_estimation_2D[:,1]
m_H_GD = np.sqrt(β_1*v_1_GD**2 + β_2*v_2_GD**2 + abs(β_1*v_1_GD**2 - β_2*v_2_GD**2))
m_h_GD = np.sqrt(β_1*v_1_GD**2 + β_2*v_2_GD**2 - abs(β_1*v_1_GD**2 - β_2*v_2_GD**2))
m_H_charged_GD = np.sqrt(β_3*(v_1_GD**2 + v_2_GD**2))


#GDM_2D Prediction
GDM_estimation_2D = Gradient_Descent_Momentum_2D(V, derivative_v_1, derivative_v_2, v_o, v_o,v_1_th,v_2_th)
#Separamos las funciones de onda v1 y v2 en sus vectores y gestionar mejor los datos
v_1_GDM = GDM_estimation_2D[:,0]
v_2_GDM = GDM_estimation_2D[:,1]
m_H_GDM = np.sqrt(β_1*v_1_GDM**2 + β_2*v_2_GDM**2 + abs(β_1*v_1_GDM**2 - β_2*v_2_GDM**2))
m_h_GDM = np.sqrt(β_1*v_1_GDM**2 + β_2*v_2_GDM**2 - abs(β_1*v_1_GDM**2 - β_2*v_2_GDM**2))
m_H_charged_GDM = np.sqrt(β_3*(v_1_GDM**2 + v_2_GDM**2))


NAG_estimation_2D = Nesterov_Accelerated_Gradient_2D(V, derivative_v_1, derivative_v_2, v_o, v_o,v_1_th,v_2_th)
#Separamos las funciones de onda v1 y v2 en sus vectores y gestionar mejor los datos
v_1_NAG = NAG_estimation_2D[:,0]
v_2_NAG = NAG_estimation_2D[:,1]
m_H_NAG = np.sqrt(β_1*v_1_NAG**2 + β_2*v_2_NAG**2 + abs(β_1*v_1_NAG**2 - β_2*v_2_NAG**2))
m_h_NAG = np.sqrt(β_1*v_1_NAG**2 + β_2*v_2_NAG**2 - abs(β_1*v_1_NAG**2 - β_2*v_2_NAG**2))
m_H_charged_NAG = np.sqrt(β_3*(v_1_NAG**2 + v_2_NAG**2))

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("Estimation of the two Higgs fields with each method:")
print(f"Gradient Descent: v_1 = {v_1_GD[-1]}, v_2 = {v_2_GD[-1]} ")
print(f"Gradient Descent with Momentum: v_1 = {v_1_GDM[-1]}, v_2 = {v_2_GDM[-1]} ")
print(f"Nesterov Accelerated Gradient: v_1 = {v_1_NAG[-1]}, v_2 = {v_2_NAG[-1]} ")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("Estimation of each Higgs boson mass with each method")
print(f"Gradient Descent: m_H = {m_H_GD[-1]}, m_h = {m_h_GD[-1]}, m_H_charged = {m_H_charged_GD[-1]} ")
print(f"Gradient Descent with Momentum: m_H = {m_H_GDM[-1]}, m_h = {m_h_GDM[-1]}, m_H_charged = {m_H_charged_GDM[-1]} ")
print(f"Nesterov Accelerated Gradient:  m_H = {m_H_NAG[-1]}, m_h = {m_h_NAG[-1]}, m_H_charged = {m_H_charged_NAG[-1]} ")


plt.plot(range(0, len(m_H_GD)), m_H_GD,label='Mass estimation with GD', color='red')
plt.plot(range(0, len(m_H_GDM)), m_H_GDM,label='Mass estimation with GDM', color='blue')
plt.plot(range(0, len(m_H_NAG)), m_H_NAG,label='Mass estimation with NAG', color='green')
plt.axhline(y=m_H_th, color='black', linestyle='--',label='Theoretical Higgs Mass')
plt.xlabel("Iterations")
plt.ylabel("Mass estimation")
plt.title("Evolution of Mass estimation of a CP-even heavier Higgs boson with each model iterations")
plt.legend()
plt.show()

plt.plot(range(0, len(m_h_GD)), m_h_GD,label='Mass estimation with GD', color='red')
plt.plot(range(0, len(m_h_GDM)), m_h_GDM,label='Mass estimation with GDM', color='blue')
plt.plot(range(0, len(m_h_NAG)), m_h_NAG,label='Mass estimation with NAG', color='green')
plt.axhline(y=m_h_th, color='black', linestyle='--',label='Theoretical Higgs Mass')
plt.xlabel("Iterations")
plt.ylabel("Mass estimation")
plt.title("Evolution of Mass estimation of a CP-even lighter Higgs boson with each model iterations")
plt.legend()
plt.show()

plt.plot(range(0, len(m_H_charged_GD)), m_H_charged_GD,label='Mass estimation with GD', color='red')
plt.plot(range(0, len(m_H_charged_GDM)), m_H_charged_GDM,label='Mass estimation with GDM', color='blue')
plt.plot(range(0, len(m_H_charged_NAG)), m_H_charged_NAG,label='Mass estimation with NAG', color='green')
plt.axhline(y=m_H_charged_th, color='black', linestyle='--',label='Theoretical Higgs Mass')
plt.xlabel("Iterations")
plt.ylabel("Mass estimation")
plt.title("Evolution of Mass estimation of a charged Higgs boson with each model iterations")
plt.legend()
plt.show()

#Second Derivative Test
def Second_Derivative_Test_1D(f,estimation,φ):
    f_xx = sp.lambdify(φ,sp.diff(f,φ,φ))
    
    if f_xx(estimation) > 0:
        print(f"  The field value {estimation} is a minimum")
    if f_xx(estimation) < 0 :
        print(f"  The field value {estimation} is a maximum")
    if f_xx(estimation) == 0 :
        print(f"  The field value {estimation} is a inflection point")

    
def Second_Derivative_Test_2D (f,v_1,v_2,v_1_model,v_2_model):
    f_xx = sp.lambdify((v_1,v_2),sp.diff(f,v_1,v_1)) #Calculates the second derivative in v_1
    det =  sp.diff(f, v_1,v_1)*sp.diff(f, v_2,v_2) -  sp.diff(f, v_2,v_1)* sp.diff(f, v_1,v_2) #Calculates the determinant of H
    Hessian_det =sp.lambdify((v_1,v_2),det) #Evaluates the values obtained with the model in the determinant
    
    if Hessian_det(v_1_model,v_2_model) > 0 : #Condition of critic point
       if f_xx(v_1_model,v_2_model) > 0:
           print(f"  The Higgs field values: {v_1_model} and {v_2_model} are minimum of the potential ")
       if f_xx(v_1_model,v_2_model) < 0:
           print(f"  The values of the Higgs field:{v_1_model} and {v_2_model} are maximum of the potential ")
    if Hessian_det(v_1_model,v_2_model) < 0:#Condition of saddle point
        print(f"  The Higgs field values: {v_1_model} and {v_2_model} are saddle points ")
    if Hessian_det(v_1_model,v_2_model) == 0: #Condition of inconclusive
        print("  The test is inconclusive")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

print("Here it will show the Second Derivative Test for 1D Higgs potential")
print("  Second Derivative test for the theoretical value:")
Second_Derivative_Test_1D(f_symb, φ_th, φ)

print("  Second Derivative test for the Gradient Descent value:")
Second_Derivative_Test_1D(f_symb, GD_estimation[-1], φ)

print("  Second Derivative test for the Gradient Descent with Momentum value:")
Second_Derivative_Test_1D(f_symb, GDM_estimation[-1], φ)

print("  Second Derivative test for the Nesterov Accelerated Gradient value:")
Second_Derivative_Test_1D(f_symb, NAG_estimation[-1], φ)
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

print("Here it will show the Second Derivative Test for the Two Higgs potential")
print("  Second Derivative test for the theoretical values:")
Second_Derivative_Test_2D(V_symb, v_1, v_2, v_1_th, v_2_th)

print("  Second Derivative test for the Gradient Descent values:")
Second_Derivative_Test_2D(V_symb, v_1, v_2, v_1_GD[-1], v_2_GD[-1])

print("  Second Derivative test for the Gradient Descent with Momentum values:")
Second_Derivative_Test_2D(V_symb, v_1, v_2, v_1_GDM[-1], v_2_GDM[-1])

print("  Second Derivative test for the Nesterov Accelerated Gradient values:")
Second_Derivative_Test_2D(V_symb, v_1, v_2, v_1_NAG[-1], v_2_NAG[-1])

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

