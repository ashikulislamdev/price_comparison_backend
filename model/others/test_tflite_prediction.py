import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Load your TFLite model ---
interpreter = tf.lite.Interpreter(model_path="model/mobilenetv2_fashion_mnist.tflite")
interpreter.allocate_tensors()

# --- Get input and output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Load a sample image ---
(_, _), (x_test, y_test) = fashion_mnist.load_data()
sample_image = x_test[0]  # Take the first test image
true_label = y_test[0]

# --- Preprocess the image ---
sample_image = tf.expand_dims(sample_image, axis=-1)  # (28,28,1)
sample_image = tf.image.resize(sample_image, (96, 96))  # (96,96,1)
sample_image = tf.image.grayscale_to_rgb(sample_image)  # (96,96,3)
sample_image = preprocess_input(sample_image)  # Normalize
sample_image = tf.expand_dims(sample_image, axis=0)  # (1,96,96,3)

# --- Set the input tensor ---
interpreter.set_tensor(input_details[0]['index'], sample_image.numpy())

# --- Run the inference ---
interpreter.invoke()

# --- Get prediction ---
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

print(f"TFLite Model Prediction: {predicted_class}")
print(f"True Label: {true_label}")
