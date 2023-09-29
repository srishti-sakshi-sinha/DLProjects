# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Create dummy variables
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X["Gender"], drop_first=True)

# Concatenate the data frames

X = pd.concat([X, geography, gender], axis=1)

# Drop Unnecessary columns

X = X.drop(['Geography', 'Gender'], axis = 1)

# Splitting the dataset into the training and testing data sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2

# Importing the Keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# Initializing the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))

# Adding the second hidden layer

classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

# Adding the output layer

classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# Compiling the ANN

classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the ANN to the Training set

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)

# list all data in history
print(model_history.history.keys())

# summarize history for accuracy

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Part 3 - Making the predictions and evaluating the model

# predicting the test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the Accuracy

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)































