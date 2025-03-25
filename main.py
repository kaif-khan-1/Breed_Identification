import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import gdown
import json

# ✅ Disable GPU for Compatibility on Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Define Model Path & Google Drive File ID
MODEL_PATH = "dog_breed_classifier.h5"  # Change to .h5 if needed
GOOGLE_DRIVE_FILE_ID = "1B79Kb1IbqeYp7GJeBJm7KVDF1QYQ0xOK"  # Replace with actual Google Drive File ID

# ✅ Download model if missing
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# ✅ Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found. Check deployment!")

# ✅ Load the Model
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ✅ Define Image Size
IMAGE_SIZE = (224, 224)

# ✅ FastAPI App
app = FastAPI(title="Dog Breed Classification API", description="Upload an image to predict the dog breed.")

@app.get("/")
def home():
    return {"message": "Welcome to the Dog Breed Classification API!"}

# ✅ Image Preprocessing (Convert UploadFile to Model Input)
def preprocess_image(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize (0-1 range)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ✅ Load Class Labels from JSON
LABELS_PATH = "class_labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as file:
        CLASS_LABELS = json.load(file)
else:
    CLASS_LABELS = [f"Breed {i}" for i in range(8)]  # Fallback labels

def predict_breed(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # ✅ Correctly map prediction to breed name
    predicted_breed = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"

    return predicted_breed, float(np.max(predictions))

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file)
    predicted_breed, confidence = predict_breed(img_array)

    return {"predicted_breed": predicted_breed, "confidence": confidence}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






