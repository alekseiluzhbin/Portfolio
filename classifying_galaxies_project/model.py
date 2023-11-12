import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

input_data, labels = load_galaxy_data()
# We set stratify = labels. This ensures that ratios of galaxies in the testing data
# will be the same as in the original dataset:
x_train, x_valid, y_train, y_valid =\
train_test_split(input_data, labels, test_size = 0.20, stratify = labels, shuffle = True, random_state = 222)

# Preprocessing image data.
# The object will normalize the pixels using the rescale parameter:
data_generator = ImageDataGenerator(rescale = 1. / 255)
# We create a training data iterator on the training data and labels:
training_iterator = data_generator.flow(x_train, y_train, batch_size = 5)
# We create a validation data iterator on the testing data and labels:
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size = 5)

# Designing the model.
# The input shape should be (128, 128, 3). This is because the images are
# 128 pixels tall, 128 pixels wide, and have 3 channels (RGB).
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (128, 128, 3)))
# Stacking convolutional layers.
# Two convolutional layers, interspersed with max pooling layers, followed by two dense layers:
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = "relu")) # 8 filters, each 3x3 with strides of 2
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(tf.keras.layers.Flatten()) # flatten layer
model.add(tf.keras.layers.Dense(16, activation = "relu")) # hidden dense layer with 16 hidden units
# outputs 4 features for the four classes ("Normal", "Ringed", "Merger", "Other"):
model.add(tf.keras.layers.Dense(4, activation = "softmax"))
# At this point the model should have 7,164 parameters:
print(model.summary())

# We compile the model with an Adam optimizer, Categorical Cross Entropy Loss, and Accuracy and AUC metrics:
model.compile(
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
	loss = tf.keras.losses.CategoricalCrossentropy(),
	metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)
model.fit(
	training_iterator,
	steps_per_epoch = len(x_train) / 5, # divided by the batch size
	epochs = 8,
	validation_data = validation_iterator,
	validation_steps = len(x_valid) / 5
)
# loss: 0.9526 - categorical_accuracy: 0.6161 - auc: 0.8416
# val_loss: 0.9324 - val_categorical_accuracy: 0.6643 - val_auc: 0.8539
