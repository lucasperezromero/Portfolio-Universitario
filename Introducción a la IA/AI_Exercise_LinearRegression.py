import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error   
from sklearn.model_selection import GridSearchCV

#Excercise###############################################################################
#  1.Generate Data following (n=1000)                                                   #
#    y = 4 + 3x + N(0,sigma^2 )                                                         #
#  2.Divide the data into training and test data. For instance, 95% and 5% respectively #
#  3.Apply OLS, Ridge and LASSO models                                                  #
#                                                                                       #
#########################################################################################

def R2(f_train, f_test, f_train_pred, f_test_pred):
    
   R_2_test = 1 - np.sum((f_test - f_test_pred)**2) / np.sum((f_test - np.mean(f_test))**2)
   R_2_training = 1 - np.sum((f_train - f_train_pred)**2) / np.sum((f_train - np.mean(f_train))**2)
   
   return R_2_test, R_2_training

#Defino mi función y sus variables 
N = 1000
sigma = 1
x = np.linspace(0 , 10, N)
noise = np.random.normal(0 , sigma**2,size= N) #Ruido Gaussiano
f = 4 + 3*x + noise #Función
x_train,x_test,f_train,f_test = train_test_split(x,f,test_size= 0.05,random_state=42 )

#Modelo Regresión Lineal
def Linear_Regression_Model(x_train,x_test,f_train,f_test):
    x_train = x_train.reshape(-1 , 1)
    x_test= x_test.reshape( -1 , 1)
    
    linear = LinearRegression()
    linear.fit(x_train , f_train)
    f_train_linear = linear.predict(x_train)
    f_test_linear = linear.predict(x_test)
    R_2_test , R_2_training = R2(f_train, f_test, f_train_linear,f_test_linear)
    return print(f"Los coeficientes de determinación de los datos de test y entrenamiento son respectivamente: {R_2_test} y {R_2_training} para la regresión Lineal")

#Modelo Regresión Ridge
def Ridge_Model(x_train,x_test,f_train,f_test):
    x_train = x_train.reshape(-1 , 1)
    x_test= x_test.reshape( -1 , 1) 
    ridge = GridSearchCV(estimator= Ridge(), param_grid= {'alpha': [0.01, 0.1, 1, 10, 100]}, scoring='neg_mean_squared_error', cv=5)
    ridge.fit(x_train, f_train)
    f_train_ridge = ridge.predict(x_train)
    f_test_ridge = ridge.predict(x_test)
    R_2_test, R_2_training = R2(f_train, f_test, f_train_ridge,f_test_ridge)
    return print(f"Los coeficientes de determinación de los datos de test y entrenamiento son respectivamente: {R_2_test} y {R_2_training} para la regresión Ridge")

Linear_Regression_Model(x_train, x_test, f_train, f_test)
Ridge_Model(x_train, x_test, f_train, f_test)
