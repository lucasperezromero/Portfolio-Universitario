from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd 
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns= data.feature_names)
df['target'] = data.target
X = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Random_Forest (x_train,x_test,y_train,y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_RF = accuracy_score(y_train, y_train_pred)
    accuracy_test_RF = accuracy_score(y_test, y_test_pred)
    
    precision_test_RF = precision_score(y_test, y_test_pred)
    precision_train_RF = precision_score(y_train, y_train_pred)
    
    return accuracy_train_RF, accuracy_test_RF, precision_train_RF, precision_test_RF

def Gradient_Boosting (x_train,x_test,y_train,y_test):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_GB = accuracy_score(y_train, y_train_pred)
    accuracy_test_GB = accuracy_score(y_test, y_test_pred)
    
    precision_test_GB = precision_score(y_test, y_test_pred)
    precision_train_GB = precision_score(y_train, y_train_pred)
    
    return accuracy_train_GB,accuracy_test_GB, precision_train_GB, precision_test_GB

def AdaBoost (x_train,x_test,y_train,y_test):
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_Ada = accuracy_score(y_train, y_train_pred)
    accuracy_test_Ada = accuracy_score(y_test, y_test_pred)
    
    precision_test_Ada = precision_score(y_test, y_test_pred)
    precision_train_Ada = precision_score(y_train, y_train_pred)
    
    return accuracy_train_Ada, accuracy_test_Ada,precision_train_Ada, precision_test_Ada


accuracy_train_RF, accuracy_test_RF, precision_train_RF, precision_test_RF = Random_Forest(x_train, x_test, y_train, y_test)
print('The accuracy of the Random Forest prediction test is:',accuracy_test_RF )
print('The accuracy of the Random Forest prediction trainning is:',accuracy_train_RF )

print('The precision of the Random Forest prediction test is:',precision_test_RF )
print('The precision of the Random Forest prediction train is:',precision_train_RF )


accuracy_train_GB, accuracy_test_GB, precision_train_GB, precision_test_GB = Gradient_Boosting(x_train, x_test, y_train, y_test)
print('The accuracy of the Gradient Boosting prediction test is:',accuracy_test_GB )
print('The accuracy of the Gradient Boosting prediction trainning is:',accuracy_train_GB )

print('The precision of the Gradient Boosting prediction test is:',precision_test_GB )
print('The precision of the Gradient Boosting prediction train is:',precision_train_GB )

accuracy_train_Ada, accuracy_test_Ada,precision_train_Ada, precision_test_Ada = AdaBoost(x_train, x_test, y_train, y_test)
print('The accuracy of the AdaBoost prediction test is:',accuracy_test_Ada )
print('The accuracy of the AdaBoost prediction trainning is:',accuracy_train_Ada )

print('The precision of the AdaBoost prediction test is:',precision_test_Ada )
print('The precision of the AdaBoost prediction train is:',precision_train_Ada )


Accuracy_comparation = [accuracy_test_RF, accuracy_test_GB, accuracy_test_Ada]

plt.bar(["Random Forest", 'Gradient Boosting', 'Adaboost'], Accuracy_comparation)
plt.xlabel("Ensamble Model")
plt.ylabel("Model Accuracy of the test models")
plt.ylim(0.8, 1.0)
plt.show()

Precision_comparation = [precision_test_RF, precision_test_GB, precision_test_Ada]

plt.bar(["Random Forest", 'Gradient Boosting', 'Adaboost'], Precision_comparation)
plt.xlabel("Ensamble Model")
plt.ylabel("Model Precision of the test models")
plt.ylim(0.8, 1.0)
plt.show()