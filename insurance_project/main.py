from tensorflow.keras.callbacks import EarlyStopping
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from model import features_train, labels_train, features_test, labels_test, design_model

# We build and train the model, creating an EarlyStopping callback
# and adding it as a parameter when we fit our model:
def fit_model(f_train, l_train, learning_rate, num_epochs):
    model = design_model(features_train, learning_rate)
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)
    history = model.fit(
        features_train, labels_train,
        epochs = num_epochs,
        batch_size = 16,
        verbose = 1,
        validation_split = 0.2,
        callbacks = [es]
    )
    return history

learning_rate = 0.1
num_epochs = 100
history = fit_model(features_train, labels_train, learning_rate, num_epochs)
print(f"Final training MAE: {history.history['mae'][-1]}")
print(f"Final validation MAE: {history.history['val_mae'][-1]}")

# Baseline
dummy_regr = DummyRegressor(strategy = 'mean')
dummy_regr.fit(features_train, labels_train)
y_pred = dummy_regr.predict(features_test)
MAE_baseline = mean_absolute_error(labels_test, y_pred)
print(f"MAE baseline: {MAE_baseline}")

