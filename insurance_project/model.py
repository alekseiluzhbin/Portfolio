import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# For reproducibility of result
# we always use the same seed for random number generator:
tf.random.set_seed(42)

# To improve readability we design the model in a separate function.
# In the case of regression, the most often used loss function is the Mean Squared Error, mse.
# Additionally, we want to observe the progress of the Mean Absolute Error, mae.
def design_model(features, learning_rate):
	model = Sequential() # initializes a Sequential model instance
	input = tf.keras.Input(shape = (features.shape[1],)) # initializes an input layer
	model.add(input) # adds input layer to a model instance
	model.add(layers.Dense(128, activation = 'relu')) # a hidden layer with 128 neurons
	model.add(layers.Dense(64, activation = 'relu'))
	model.add(layers.Dense(24, activation = 'relu'))
	model.add(layers.Dense(1)) # 1 neuron in the output layer
	opt = tf.keras.optimizers.Adam(learning_rate = learning_rate) # the Adam optimizer
	model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
	return model

# We load and split the data into features and labels:
dataset = pd.read_csv('insurance.csv')
features = dataset.iloc[:, 0:6]
labels = dataset.iloc[:, -1]

# Data preprocessing: one-hot encoding and standardization.
# One-hot encoding creates a binary column for each category:
features = pd.get_dummies(features)
# split data into train and test sets:
features_train, features_test, labels_train, labels_test =\
train_test_split(features, labels, test_size = 0.33, random_state = 42)
# Standardization rescales features to zero mean and unit variance:
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder = 'passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)