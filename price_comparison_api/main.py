from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or device IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model

model_path = os.path.join(os.path.dirname(__file__), "mobilenetv2_fashion_mnist.tflite")
model = tf.lite.Interpreter(model_path=model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"TFLite model not found at {model_path}")
# else:
#     print(f"Model loaded from {model_path}")


model.allocate_tensors()

# Load labels
# with open("labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

with open(os.path.join(os.path.dirname(__file__), "labels.txt"), "r") as f:
    labels = [line.strip() for line in f.readlines()]
if not labels:
    raise ValueError("Labels file is empty or not found.")

# Preprocess function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((96, 96))  # model's input size
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)

        input_index = model.get_input_details()[0]["index"]
        output_index = model.get_output_details()[0]["index"]

        model.set_tensor(input_index, input_data)
        model.invoke()
        output_data = model.get_tensor(output_index)

        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        return JSONResponse(content={
            "predicted_class": labels[predicted_index],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
