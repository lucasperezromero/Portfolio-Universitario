from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = load_digits() #Asigna los datos de los digitos a una variable
df = pd.DataFrame(data.data, columns= data.feature_names) #Crea el data frame con los datos
df['target'] = data.target #Incluye la columna "target" que determina el digito en función de los pixeles
X = df.drop('target', axis=1) #Asigno los datos de los pixeles como la x
y = df['target'] #Asigno tipo de digito a la y

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Hace el split de los datos de entrenamiento y los datos de testeo

def Random_Forest (x_train,x_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators=50) #Asigno el Random Forest como el modelo con 50 arboles 
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train) #Predice el digito en base a los datos de entrenamiento
    y_test_pred = model.predict(x_test)#Predice el digito en base a los datos de testeo 
    
    accuracy_train_RF = accuracy_score(y_train, y_train_pred) #Calcula la accuracy del modelo de entrenamiento
    accuracy_test_RF = accuracy_score(y_test, y_test_pred) #Calcula la accuracy del modelo de test
    return accuracy_test_RF, accuracy_train_RF

def Gradient_Boosting (x_train,x_test,y_train,y_test):
    model = GradientBoostingClassifier(n_estimators=50) #Asigno Gradient Boosting como el modelo
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train) #Predicción con los datos de entreno 
    y_test_pred = model.predict(x_test) #Predicción con los datos de test
    
    accuracy_train_GB = accuracy_score(y_train, y_train_pred) #Accuracy del entrenamiento
    accuracy_test_GB = accuracy_score(y_test, y_test_pred) #Accuracy del test
    
    return accuracy_test_GB, accuracy_train_GB

def AdaBoost (x_train,x_test,y_train,y_test):
    model = AdaBoostClassifier(n_estimators=50) #Asigno Ada Boost como el modelo a entrenar
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train) #Predicción con los datos de entreno
    y_test_pred = model.predict(x_test) #Predicción con los datos de test
    
    accuracy_test_AB = accuracy_score(y_test, y_test_pred) #Accuracy del test
    accuracy_train_AB = accuracy_score(y_train, y_train_pred) #Accuracy del entrenamiento
    
    return accuracy_test_AB, accuracy_train_AB

#Llamo a las funciones y reporto las accuracies 
accuracy_test_RF, accuracy_train_RF = Random_Forest(x_train, x_test, y_train, y_test)
print('The accuracy of the Random Forest prediction test is:',accuracy_test_RF )


accuracy_test_GB, accuracy_train_GB = Gradient_Boosting(x_train, x_test, y_train, y_test)
print('The accuracy of the Gradient Boosting prediction test is:',accuracy_test_GB )


accuracy_test_Ada, accuracy_train_Ada = AdaBoost(x_train, x_test, y_train, y_test)
print('The accuracy of the AdaBoost prediction test is:',accuracy_test_Ada )

Accuracy_comparation = [accuracy_test_RF, accuracy_test_GB, accuracy_test_Ada]

#Histograma de la accuracy de los modelos con los datos de test
plt.bar(["Random Forest", 'Gradient Boosting', 'Adaboost'], Accuracy_comparation)
plt.xlabel("Ensamble Model")
plt.ylabel("Test Set Model Accuracy")
plt.ylim(0.8, 1.0)
plt.show()

Accuracy_comparation = [accuracy_train_RF, accuracy_train_GB, accuracy_train_Ada]
#Histograma de la accuracy de los modelos con los datos de entrenamiento
plt.bar(["Random Forest", 'Gradient Boosting', 'Adaboost'], Accuracy_comparation)
plt.xlabel("Ensamble Model")
plt.ylabel("Train Set Model Accuracy")
plt.ylim(0.8, 1.0)
plt.show()

#Discusión de los resultados:
    #El modelo que mejor resultados da es el Random Forest con una accuracy de 0.97. Posiblemente por no depender del gradiente de los errores, como el Gradient Boosting ya que puede haber un gradient vanish y el Ada Boost tiene aplicación en clases binarias (1 o 0) pero en el caso de los digitos las clases son de 1 a 9 por lo que los predictores no los puede ajustar adecuadamente 

