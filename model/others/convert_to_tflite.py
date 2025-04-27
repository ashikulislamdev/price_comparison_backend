import tensorflow as tf

# Load your trained Keras .h5 model
model = tf.keras.models.load_model('model/mobilenetv2_fashion_mnist.h5')

# Set up the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# optimizations (for smaller and faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

with open('model/mobilenetv2_fashion_mnist.tflite', 'wb') as f:
    f.write(tflite_model)

print("Saved as mobilenetv2_fashion_mnist.tflite")
