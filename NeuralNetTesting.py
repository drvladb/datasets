# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('TestData.csv')
print(dataset)
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values
print(X)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train)
print(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, init = 'uniform', activation = 'relu', input_dim = 7))
# Adding the second hidden layer
classifier.add(Dense(6, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 5)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

classifier.evaluate

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# new instance where we do not know the answer
from numpy import array
Xnew = array([[2, 4, 1], [4, 2, 1], [4, 1, 0]])
# make a prediction
ynew = classifier.predict_classes(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))