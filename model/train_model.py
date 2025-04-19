import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess: Resize to 224x224x3 to match VGG16 input and convert grayscale to RGB
# def preprocess_images(images):
#     resized = []
#     for img in images:
#         img = tf.image.resize(img[..., np.newaxis], (224, 224))        # Resize to 224x224
#         img = tf.image.grayscale_to_rgb(img)                            
#         resized.append(img.numpy())
#     return np.array(resized) / 255.0  

# def preprocess_images(images):
#     images = tf.expand_dims(images, axis=-1)  # Add channel dimension
#     images = tf.image.resize(images, (224, 224))  # Resize all at once
#     images = tf.image.grayscale_to_rgb(images)    # Convert to 3 channels
#     return images.numpy()  # Only convert once at the end

def preprocess_images(images):
    images = tf.expand_dims(images, axis=-1)  # Make sure it has channel dimension
    images = tf.image.resize(images, (224, 224))
    images = images / 255.0
    return images


x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Build the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model
model.save("fashion_model.h5")
print("Model training complete and saved as fashion_model.h5")
