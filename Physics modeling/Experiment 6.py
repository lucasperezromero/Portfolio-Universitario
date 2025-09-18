import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = [37.7051e6, 40.015e6,45.012e6,50.06e6,55.055e6,60.003e6,65.009e6,70.009e6] #Frequency Hz
U_r = [0.5,0.72,1.26,1.7,2.18,2.64,3.04,3.46] #Potential V
B= [] #Empty Magnetic Field Array
g = [] #Empty Lande constants array
u_B = 9.274e-24 #J/T
h = 6.62607e-34# Js
B_o = 0.347*1e-3 #T/V factor conversión a campo magnetico


for i in range(len(U_r)):
    B.append(B_o*U_r[i])
    g.append((h/u_B)*(f[i]/B[i]))

df = pd.DataFrame({'f (Hz)': f,'U_r (V)': U_r,'B (T)': B,'g': g})
print(df)

coeff = np.polyfit(B,f, 1)
a = coeff[0]
g_est = a*h/u_B
print(f"El factor de Lande es {g_est}")



f_fit = np.poly1d(coeff)(B)

plt.plot(B, f, 'o', label='Datos')
plt.plot(B, f_fit, '-', label='Ajuste lineal')
plt.xlabel('Campo magnético B (T)')
plt.ylabel('Frecuencia f (Hz)')
plt.title("Frecuencia vs Campo magnético")
plt.legend()
plt.grid(True)
plt.show()



