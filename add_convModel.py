from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models
import numpy as np
import pickle

# Load additional data
with open("saved_model/features.pkl", "rb") as file:
    add_features = pickle.load(file)

with open("saved_model/targets.pkl", "rb") as file:
    add_targets = pickle.load(file)

# Load mnist data from keras datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate our additional data
new_features = add_features[:,:,:, 0]
x_train = np.concatenate([x_train, new_features])

y_train = np.concatenate([y_train, add_targets])

# Reshape and normalize image data
train_images = x_train.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = x_test.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_label = to_categorical(y_train)
test_label = to_categorical(y_test)

# Build convolutional model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding classifier on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_label, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_label)
print(test_acc)

# Save model
with open("saved_model/add_conv_model.pkl", 'wb') as file:
    saved = pickle.dump(model, file)
