import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical

# Loading the data to a pandas DataFrame object:
data = pd.read_csv("dataset.csv")
# Print all the columns and their types:
print(data.info())
# Print the distribution of the death_event column (the column we need to predict):
print(Counter(data.DEATH_EVENT))
y = data.iloc[:, -1] # extract the label column death_event
x = data.iloc[:, :-1] # extract the features columns

# Data preprocessing.
# Converting the categorical features to one-hot encoding vectors:
x = pd.get_dummies(x)
# Splitting the data into training features, test features, training labels, and test labels, respectively:
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
numeric_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
# Initializing a ColumnTransformer object to scale the numeric features in the dataset:
ct = ColumnTransformer([("only numeric", StandardScaler(), numeric_features)], remainder = 'passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Prepare labels for classification.
# The following code initializes an instance of LabelEncoder
# to encode the labels into integers:
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype('str'))
Y_test = le.transform(Y_test.astype('str'))
# The following code transforms the encoded labels into a binary vector:
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Design the model.
model = Sequential()
model.add(InputLayer(input_shape = (X_train.shape[1],)))
model.add(Dense(12, activation = 'relu')) # a hidden layer with 12 neurons
model.add(Dense(2, activation = 'softmax')) # an output layer with the number of neurons corresponding to the number of classes
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Train and evaluate the model.
model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose = 0)
loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
print(f"Loss: {loss}", f"Accuracy: {acc}")

# Generating a classification report.
y_estimate = model.predict(X_test, verbose = 0)
y_estimate = np.argmax(y_estimate, axis = 1) # to select the indices of the true classes for each label encoding
y_true = np.argmax(Y_test, axis = 1)
print(classification_report(y_true, y_estimate)) # to observe the F1-score amongst other metrics
