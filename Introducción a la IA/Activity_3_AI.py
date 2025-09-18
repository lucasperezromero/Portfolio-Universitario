import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


def R2(f_th ,f_pred):
    
   R_2= 1 - np.sum((f_th - f_pred)**2)/np.sum((f_th - np.mean(f_th))**2)
   
   return R_2

#Free fall equation with gaussian noise
N = 100 
sigma = 100
t = np.linspace(0 , 100, N) 
noise = np.random.normal(0 , sigma , N) #Gaussian Noise
g= 9.8 #Gravity
t_2 = t**2
y = (1/2)*g*t_2 + noise #Function

x_train, x_test, f_train, f_test = train_test_split(t_2, y, test_size=0.1, random_state=42)

x_train = x_train.reshape(-1 , 1)
x_test = x_test.reshape( -1 , 1)
f_train = f_train.ravel()
f_test = f_test.ravel()


# Add a column of ones for the intercept term
x_train_b = np.c_[np.ones((x_train.shape[0], 1)), x_train]
x_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]

#Linear Regression
w_ols = np.linalg.inv(x_train_b.T@x_train_b)@(x_train_b.T@f_train)

print(f"Los coeficientes w obtenidos mediante OLS son: {w_ols} ")

f_train_linear = x_train_b@w_ols
f_test_linear = x_test_b@w_ols
R_2_test_linear = R2(f_test, f_test_linear)
R_2_training_linear = R2(f_train, f_train_linear)
print(f"Los coeficientes de determinación R^2 de los datos de test y entrenamiento con la Linear Regression son respectivamente: {R_2_test_linear} y {R_2_training_linear}")


#Ridge Regression
I = np.eye(x_train_b.shape[1])  # Matriz identidad
alpha = np.logspace(-4, 0, 10)
w_ridge= []
R_2_test_ridge = []
R_2_training_ridge = []

for alph in alpha: 
    w_ridge.append(np.linalg.inv(x_train_b.T@x_train_b + alph*I)@(x_train_b.T@f_train))
    
    f_train_ridge = x_train_b@w_ridge[-1]
    f_test_ridge = x_test_b@w_ridge[-1]
    
    R_2_test_ridge.append(R2(f_test,f_test_ridge))
    R_2_training_ridge.append(R2(f_train, f_train_ridge))
    
best_R2_training_ridge = np.argmax(R_2_training_ridge)
best_R2_test_ridge = np.argmax(R_2_test_ridge)
print(f"Los coeficientes w obtenidos mediante Ridge Regression son: {w_ridge} ")

print(f"Los coeficientes de determinación R^2 de los datos de test y entrenamiento con la Ridge Regression son respectivamente:{R_2_test_ridge[best_R2_test_ridge]:.5f} y {R_2_training_ridge[best_R2_training_ridge]:.5f}")


#Lasso Regression
R_2_test_lasso = []
R_2_training_lasso = []
w_lasso = []

for alph in alpha: 
    lasso = Lasso(alpha= alph, max_iter= 5000, fit_intercept=True)
    lasso.fit(x_train, f_train)
    
    f_train_lasso = lasso.predict(x_train)
    f_test_lasso = lasso.predict(x_test)
    w_lasso.append(lasso.coef_)
    
    R_2_test_lasso.append(R2(f_test,f_test_lasso))
    R_2_training_lasso.append(R2(f_train,f_train_lasso))

best_R2_training_lasso = np.argmax(R_2_training_lasso)
best_R2_test_lasso = np.argmax(R_2_test_lasso)

print(f"Los coeficientes w obtenidos mediante Lasso Regression son: {w_lasso} ")
print(f"Los coeficientes de determinación R^2 de los datos de test y entrenamiento con la Lasso Regression son respectivamente: {R_2_test_lasso[best_R2_test_lasso]:.5f} y {R_2_training_lasso[best_R2_training_lasso]:.5f}")


plt.plot(t_2, y, label="Datos simulados",linestyle='dashed', color="gray", alpha=.5)
plt.plot(x_train, f_train_linear, label="Linear Regression Trainning", linestyle="dashed", linewidth=2)
plt.plot(x_test,f_test_linear,label="Linear Regression Test", linestyle="dashed", linewidth=2 )
plt.plot(x_train, f_train_ridge, label="Ridge Regression Trainning", linestyle="dashed", linewidth=2)
plt.plot(x_test, f_test_ridge, label="Ridge Regression Test", linestyle="dashed", linewidth=2)
plt.plot(x_train, f_train_lasso, label="Lasso Regression Trainning", linestyle="dashed", linewidth=2)
plt.plot(x_test,f_test_lasso, label="Lasso Regression Test", linestyle = "dashed", linewidth=2)
plt.xlabel("$t^2$")
plt.ylabel("$y$")
plt.legend()
plt.title("Regresión en caída libre con ruido")
plt.show()

# Gráfico de coeficientes vs alpha
plt.figure(figsize=(10, 4))
plt.plot(alpha, np.array(w_ridge)[:,1], marker="o", label="Ridge Coefficients")
plt.plot(alpha, np.array(w_lasso), marker="s", label="Lasso Coefficients")
plt.xlabel("Alpha")
plt.ylabel("Coeficientes")
plt.legend()
plt.title("Coeficientes vs Regularización")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(alpha, R_2_training_ridge, marker="o", label="Ridge $R^2$ for model trainning")
plt.plot(alpha, R_2_test_ridge, marker="s", label="Ridge $R^2$ for model test")
plt.plot(alpha, R_2_training_lasso, marker="o", label="Lasso $R^2$ for model trainning")
plt.plot(alpha, R_2_test_lasso, marker="s", label="Lasso $R^2$ for model test")
plt.xlabel("Alpha")
plt.ylabel("$R^2$")
plt.legend()
plt.title("Coeficientes vs $R^2$")
plt.show()


