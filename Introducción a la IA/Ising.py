import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

L = 10  #Lattice length
steps = 100 #Metropoli steps


i = np.random.randint(0,L)
j = np.random.randint(0,L)
T = np.linspace(1.0, 3.5,num= 100)
T = np.append(T, 2.269) #Includes critical temperature
T = np.sort(T)  #Sorts the array so T_c its in the right position in relation to the other temperatures
configs = []
temps = []
phase = []


for t in T:
    
    lattice = np.random.choice([-1,+1],size = [L, L]) #Creates a matrix of random spin +1/-1
    for step in range(steps):
        for a in range(L*L):  # L^2 random flips
           
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            dE = 2*lattice[i,j]*np.sum([lattice[(i+1)%L , j], lattice[(i-1)%L,j], lattice[i,(j+1)%L], lattice[i,(j-1)%L]]) #Energy difference
           
            if dE <= 0 : 
                
                lattice[i,j] *= -1 
                
            if dE > 0: 
                r = np.random.rand()  # Random number between 0 and 1
                if r < np.exp(-dE / t):
                    lattice[i, j] *= -1  # Probability flipping
    
    configs.append(lattice.flatten()) #Saves the new configuration of spin
    temps.append(t) #Saves the temperature of the configuration
    if t < 2.269:
        phase.append(0) #Ordered phase
    else:
        phase.append(1) #Disordered phase
    if t == 2.269:
        lattice_crit = lattice #Saves the configuration at critic temperature
    if t == T[0]:
        lattice_low = lattice #Saves the lattice configuration at low temperature
    if t == T[99]:
        lattice_high = lattice #Saves the lattice configuration at high temperature


plt.imshow(lattice_low, cmap="coolwarm")  
plt.title(f"Spin configuration at low temperature T = {T[0]}")
plt.colorbar()
plt.show()

plt.imshow(lattice_high, cmap="coolwarm")  
plt.title(f"Spin configuration at high temperature T = {T[99]}")
plt.colorbar()
plt.show()

plt.imshow(lattice_crit, cmap="coolwarm")  
plt.title("Spin configuration for the critial temperature 2.269")
plt.colorbar()
plt.show()



#Dataframe creation 
df = pd.DataFrame(configs)
df["T"] = temps
df["label"] = phase


X = df.iloc[:, :L*L]
y = df['label']
#Splitting the data in trainning and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Bagging (x_train,x_test,y_train,y_test):
    model = BaggingClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_B = accuracy_score(y_train, y_train_pred)
    accuracy_test_B = accuracy_score(y_test, y_test_pred)
    
    precision_test_B = precision_score(y_test, y_test_pred)
    precision_train_B = precision_score(y_train, y_train_pred)
    
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Data")
    plt.ylabel("True Data")
    plt.title("Bagging model - Confusion Matrix")
    plt.show()
    
    return accuracy_train_B, accuracy_test_B, precision_train_B, precision_test_B

def Gradient_Boosting (x_train,x_test,y_train,y_test):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_GB = accuracy_score(y_train, y_train_pred)
    accuracy_test_GB = accuracy_score(y_test, y_test_pred)
    
    precision_test_GB = precision_score(y_test, y_test_pred)
    precision_train_GB = precision_score(y_train, y_train_pred)
    
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Data")
    plt.ylabel("True Data")
    plt.title("Gradient Boosting - Confusion Matrix")
    plt.show()
    
    return accuracy_train_GB,accuracy_test_GB, precision_train_GB, precision_test_GB


def Random_Forest (x_train,x_test,y_train,y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    accuracy_train_RF = accuracy_score(y_train, y_train_pred)
    accuracy_test_RF = accuracy_score(y_test, y_test_pred)
    
    precision_test_RF = precision_score(y_test, y_test_pred)
    precision_train_RF = precision_score(y_train, y_train_pred)
    
    importance = model.feature_importances_
    importance_matrix = importance.reshape((L, L))

    plt.figure(figsize=(6,5))
    sns.heatmap(importance_matrix, cmap="viridis")
    plt.title("Feature importance (spins) - Random Forest")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.show()
    
    return accuracy_train_RF, accuracy_test_RF, precision_train_RF, precision_test_RF


#Random Forest results
accuracy_train_RF, accuracy_test_RF, precision_train_RF, precision_test_RF = Random_Forest(x_train, x_test, y_train, y_test)
print('The accuracy of the Random Forest prediction test is:',accuracy_test_RF )
print('The accuracy of the Random Forest prediction trainning is:',accuracy_train_RF )

print('The precision of the Random Forest prediction test is:',precision_test_RF )
print('The precision of the Random Forest prediction train is:',precision_train_RF )

#Gradient Boosting results
accuracy_train_GB, accuracy_test_GB, precision_train_GB, precision_test_GB = Gradient_Boosting(x_train, x_test, y_train, y_test)
print('The accuracy of the Gradient Boosting prediction test is:',accuracy_test_GB )
print('The accuracy of the Gradient Boosting prediction trainning is:',accuracy_train_GB )

print('The precision of the Gradient Boosting prediction test is:',precision_test_GB )
print('The precision of the Gradient Boosting prediction train is:',precision_train_GB )


#Bagging results
accuracy_train_B, accuracy_test_B, precision_train_B, precision_test_B = Bagging(x_train, x_test, y_train, y_test)
print('The accuracy of the Bagging prediction test is:',accuracy_test_B )
print('The accuracy of the Bagging prediction trainning is:',accuracy_train_B )

print('The precision of the Bagging prediction test is:',precision_test_B )
print('The precision of the Bagging prediction train is:',precision_train_B )

print("Hello World")
