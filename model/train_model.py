import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# --- Configuration ---
IMG_SIZE = 224 # VGG16 input size
BATCH_SIZE = 36 # Adjust based on your available RAM/GPU memory
BUFFER_SIZE = 1000 # For shuffling

# --- Load Raw Data ---
# Load data as numpy arrays FIRST, without initial preprocessing
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = fashion_mnist.load_data()

# --- Define Preprocessing Function (for tf.data) ---
# This function processes ONE image tensor at a time
def preprocess_image(image, label):
    # Add channel dimension
    image = tf.expand_dims(image, axis=-1)
    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Convert grayscale to RGB
    image = tf.image.grayscale_to_rgb(image)
    # Normalize to [0, 1] - VGG16 expects this range or specific preprocessing
    # Note: VGG16 often uses tf.keras.applications.vgg16.preprocess_input
    #       which converts to BGR and centers around ImageNet means.
    #       Using simple / 255.0 is okay for transfer learning but sub-optimal.
    #       Let's use the dedicated function for better results.
    # image = image / 255.0 # Simple normalization
    image = tf.keras.applications.vgg16.preprocess_input(image) # VGG16 specific preprocessing

    # One-hot encode the label
    label = tf.one_hot(label, depth=10)
    return image, label

# --- Build tf.data Pipelines ---
# Training Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_raw, y_train_raw))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache() # Cache after preprocessing (if dataset fits in memory after map, speeds up epochs)
                                       # Remove .cache() if the processed dataset is STILL too large for RAM.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Prefetch next batch

# Testing Dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_raw, y_test_raw))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE) # No shuffling for test set
# test_dataset = test_dataset.cache() # Cache test set if it fits in memory
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- Load VGG16 model ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers

# --- Build the full model ---
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# --- Compile the model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Print model summary ---
model.summary()

# --- Train the model using the datasets ---
print("Starting model training...")
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset
    # Removed batch_size here, as it's handled by the dataset pipeline
    # steps_per_epoch and validation_steps are inferred automatically from tf.data datasets
)

# --- Save the trained model ---
model.save("fashion_model.h5")
print("Model training complete and saved as fashion_model.h5")