import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

# Loading and analyzing the data.
train_data = pd.read_csv("data_train.csv")
test_data = pd.read_csv("data_test.csv")

print(train_data.info()) # print columns and their respective types
print(Counter(train_data["Air_Quality"])) # print the class distribution
x_train = train_data.iloc[:, 0:-1] # extract the features from the training data
y_train = train_data.iloc[:, -1] # extract the label column from the training data
x_test = test_data.iloc[:, 0:-1] # extract the features from the test data
y_test = test_data.iloc[:, -1] # extract the label column from the test data

# Data preprocessing.
le = LabelEncoder() # encode the labels into integers
# Each category is mapped to an integer:
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))
# We can convert labels as integers into one-hot-encodings (the format we need for cross-entropy loss):
y_train = tf.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tf.keras.utils.to_categorical(y_test, dtype = 'int64')

# Designing a deep learning model for classification.
model = Sequential()
model.add(InputLayer(input_shape = (x_train.shape[1],))) # add the input layer
model.add(Dense(10, activation = 'relu')) # add a hidden layer with 10 neurons and relu activation function.

# For classification we need to use the softmax activation function
# that outputs a vector with elements having values between 0 and 1 and that sum to 1:
model.add(Dense(6, activation = 'softmax')) # add an output layer: 6 classes in the Air Quality data.

# Compile the model.
# Cross-entropy is a score that summarizes the average difference between the actual
# and predicted probability distributions for all classes.
# The goal is to minimize the score, with a perfect cross-entropy value is 0.
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Train and evaluate the classification model.
model.fit(x_train, y_train, epochs = 20, batch_size = 16, verbose = 1)
y_estimate = model.predict(x_test)
# Convert the one-hot-encoded labels into the index of the class the sample belongs to:
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)
# F1-score is a helpful way to evaluate the model
# that takes both precision and recall into account with the harmonic mean.
# To observe the F1-score amongst other metrics:
print(classification_report(y_true, y_estimate))
