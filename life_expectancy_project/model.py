import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv('dataset.csv')
# To create a predictive model, knowing from which country data comes,
# can be confusing and it is not a column we can generalize over:
# the Country column is dropped from the DataFrame.
dataset.drop(columns = ['Country'], inplace = True)
# split the data into labels and features:
features = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]

# Data preprocessing: one-hot encoding and standardization.
# One-hot encoding creates a binary column for each category:
features = pd.get_dummies(dataset)

# split the data into training set and test sets:
features_train, features_test, labels_train, labels_test =\
train_test_split(features, labels, test_size = 0.33, random_state = 42)

# Standardization rescales numerical features to zero mean and unit variance:
numerical_features = features.select_dtypes(include = ['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([('only numeric', StandardScaler(), numerical_columns)], remainder = 'passthrough')
features_train_scaled = ct.fit_transform(features_train) # fit and transform
features_test_scaled = ct.transform(features_test) # using the trained ColumnTransformer instance ct

# Building the model:
my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1],)) # input layer
my_model.add(input)
my_model.add(Dense(64, activation = 'relu')) # hidden layer
my_model.add(Dense(1)) # output layer

# Initializing the optimizer and compiling the model:
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

# Fit and evaluate the model:
my_model.fit(features_train_scaled, labels_train, batch_size = 1, epochs = 40, verbose = 1)
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(f'Mean Squared Error: {res_mse}')
print(f'Mean Absolute Error: {res_mae}')
