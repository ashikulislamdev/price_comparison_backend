import tensorflow as tf

keras_model_filename = 'model/mobilenetv2_fashion_mnist.h5'

tflite_model_filename = 'model/mobilenetv2_fashion_mnist.tflite'

print(f"TensorFlow Version used for conversion: {tf.__version__}")

# Load model
try:
    model = tf.keras.models.load_model(keras_model_filename)
    print(f"Successfully loaded Keras model: {keras_model_filename}")
except Exception as e:
    print(f"Error loading Keras model '{keras_model_filename}': {e}")
    exit()

# convert to TensorFlow Lite 
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    tflite_model = converter.convert()
    print("Model conversion to TFLite format (NO OPTIMIZATIONS) successful.")
except Exception as e:
    print(f"Error during TFLite conversion: {e}")
    exit()

# save model
try:
    with open(tflite_model_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved as: {tflite_model_filename}")
except Exception as e:
    print(f"Error saving TFLite model: {e}")
    exit()
