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

def R_2 (y_pred , y_true):
    R_2 = 1 - sum(abs(y_true - y_pred)**2)/sum(abs(y_true - np.mean(y_pred))**2)
    return R_2

N= 1000
sigma = 1
noise = np.random.normal(0,sigma**2)
x = np.linspace(0, 10, N)
y = 4 + 3*x + noise

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size= 0.05, random_state= 42)


x_test = x_test.reshape(-1, 1)
x_train = x_train.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)

R_2_train = R_2(y_train_predict,y_train)
R_2_test = R_2(y_test_predict,y_test)

print(f"The accuracy of the training model is {R_2_train}")
print(f"The accuracy of the test model is {R_2_test}")
