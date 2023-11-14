import tensorflow as tf

# Designing the model. Stacking convolutional layers.
def design_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape = (256, 256, 1))) # the shape of the image data

    # Defines a convolutional layer with 5 filters, each of size 5 by 5:
    model.add(tf.keras.layers.Conv2D(5, 5, strides = 3, activation = "relu"))
    # Defines a max pooling layer that will move a 2x2 window across the input:
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    # Regularization: dropout (percentage of layer outputs set to zero - 10%)
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv2D(3, 3, strides = 1, activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    # to flatten image data and create the output layer:
    model.add(tf.keras.layers.Flatten())
    # three different classes - three different potential outputs:
    model.add(tf.keras.layers.Dense(3, activation = "softmax"))
    # print(model.summary())

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
    )
    return model
