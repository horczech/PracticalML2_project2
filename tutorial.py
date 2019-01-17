import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# preprocess data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create model
model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, tf.nn.relu))
model.add(tf.keras.layers.Dense(128, tf.nn.relu))

model.add(tf.keras.layers.Dense(10, tf.nn.softmax))


# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=3)

plt.imshow(x_train[0], cmap='gray')
plt.show()