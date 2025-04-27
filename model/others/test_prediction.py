from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf

# Load the saved model
model = load_model('model/mobilenetv2_fashion_mnist.h5')

# Preprocess a single new image (example)
# Assume new_image is a (28,28) numpy array (like FashionMNIST)
new_image = np.random.rand(28, 28) * 255  # dummy random image
new_image = tf.expand_dims(new_image, axis=-1) # (28, 28, 1)
new_image = tf.image.resize(new_image, (96, 96)) # (96, 96, 1)
new_image = tf.image.grayscale_to_rgb(new_image) # (96, 96, 3)
new_image = preprocess_input(new_image)  # MobileNetV2 preprocessing
new_image = tf.expand_dims(new_image, axis=0)  # Add batch dimension (1,96,96,3)

# Predict
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)
