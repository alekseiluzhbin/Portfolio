from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

BATCH_SIZE = 16
# Preprocessing image data.
# We use ImageDataGenerators to load images from a file path, and to preprocess them.
training_data_generator = ImageDataGenerator(
	# pixel normalization and data augmentation:
	rescale = 1. / 255,
	zoom_range = 0.2,
	rotation_range = 15,
	width_shift_range = 0.05,
	height_shift_range = 0.05)
training_iterator = training_data_generator.flow_from_directory(
	'data/train',
	class_mode = 'categorical',
	color_mode = 'grayscale',
	batch_size = BATCH_SIZE)
# We have to define another ImageDataGenerator and use it to load our validation data.
# Like with our training data, we are going to need to normalize our pixels.
# However, unlike for our training data, we will not augment the validation data with random shifts.
validation_data_generator = ImageDataGenerator(rescale = 1.0 / 255)
validation_iterator = validation_data_generator.flow_from_directory(
	'data/test', # to load the validation grayscale images
	class_mode = 'categorical',
	color_mode = 'grayscale',
	batch_size = BATCH_SIZE)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (256, 256, 1)))
# Stacking convolutional layers.
# Defines a convolutional layer with 2 filters, each of size 5 by 5:
model.add(tf.keras.layers.Conv2D(2, 5, strides = 3, activation = "relu"))
# Defines a max pooling layer that will move a 5x5 window across the input:
model.add(tf.keras.layers.MaxPooling2D(pool_size = (5, 5), strides = (5, 5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides = 1, activation = "relu")) 
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
# Flatten layer allows to combine the dimensions of the image into a single, lengthy feature vector.
# We can then pass this output to a Dense layer.
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
print(model.summary())

# We compile the model with an Adam optimizer, Categorical Cross Entropy Loss, and Accuracy and AUC metrics.
# Because our labels are onehot, we will use CategoricalCrossentropy as our loss function.
# Because our dateset is balanced, accuracy is a meaningful metric.
# We will also include AUC (area under the ROC curve). We want AUC to be as close to 1.0 as possible.
model.compile(
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005),
	loss = tf.keras.losses.CategoricalCrossentropy(),
	metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)
model.fit(
	training_iterator,
	steps_per_epoch = training_iterator.samples / BATCH_SIZE,
	epochs = 5,
	validation_data = validation_iterator,
	validation_steps = validation_iterator.samples / BATCH_SIZE
)
# categorical_accuracy: 0.8810 - auc: 0.9506 - val_loss: 0.4587 - val_categorical_accuracy: 0.8050 - val_auc: 0.8820