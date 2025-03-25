import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import json

# ✅ Disable GPU & OneDNN for Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Load Model (Ensure it exists)
MODEL_PATH = "model.keras"  # Change to .h5 if needed
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found. Ensure it is uploaded correctly.")

model = load_model(MODEL_PATH, compile=False)

# ✅ Load Class Labels from JSON (or Default)
LABELS_PATH = "class_labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as file:
        CLASS_LABELS = json.load(file)
else:
    CLASS_LABELS = ["Unknown"]  # Default

# ✅ FastAPI App
app = FastAPI(title="Dog Breed Classification API", description="Upload an image to predict the dog breed.")

@app.get("/")
def home():
    return {"message": "Welcome to the Dog Breed Classification API!"}

@app.get("/favicon.ico")
def favicon():
    return {}

# ✅ Preprocess Uploaded Image
def preprocess_image(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class < len(CLASS_LABELS):
        predicted_breed = CLASS_LABELS[predicted_class]
    else:
        predicted_breed = "Unknown"

    return {"predicted_breed": predicted_breed, "confidence": float(np.max(predictions))}

# ✅ Run FastAPI Server (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
