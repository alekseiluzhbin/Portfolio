import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# We create a neural network model to perform a regression analysis on the admission data.
# Since this is a regression model, the loss function should be mean-squared error (MSE),
# with mean-average error (MAE) function as the metric.
def design_model(features, learning_rate):
    model = Sequential() # initializes a Sequential model instance
    input = InputLayer(input_shape = (features.shape[1],)) # initializes an input layer
    model.add(input)
    model.add(Dense(24, activation = 'relu')) # a hidden layer with 24 neurons
    model.add(Dense(1))
    opt = Adam(learning_rate = learning_rate) # the Adam gradient descent optimizer
    model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
    return model

dataset = pd.read_csv('dataset.csv')
# We split it up the data into feature parameters and the labels:
features = dataset.iloc[:, 1:-1]
labels = dataset.iloc[:, -1]

# Data preprocessing.
# There are no categorical variables in this dataset, so we do not have to perform one-hot encoding.
# We have created two DataFrames: one for features and one for labels.
# Now we must split each of these into a training set and a test set:
features_train, features_test, labels_train, labels_test =\
train_test_split(features, labels, test_size = 0.33, random_state = 42)

# If we look through the dataset, we may notice that there are many different scales being used.
# We should either scale or normalize data so that all features have equal weight in the learning model:
ct = ColumnTransformer([('standardize', StandardScaler(), features.columns)])
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# We fit the model with the training set and test it out with the test set:
learning_rate = 0.01
num_epochs = 10
batch_size = 1
my_model = design_model(features, learning_rate)
my_model.fit(features_train_scaled, labels_train, batch_size = batch_size, epochs = num_epochs, verbose = 1)
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(f'Mean Squared Error: {res_mse}')
print(f'Mean Absolute Error: {res_mae}')
