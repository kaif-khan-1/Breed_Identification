import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import gdown

# ✅ Disable GPU & OneDNN for Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Define Model Path & Google Drive File ID
MODEL_PATH = "dog_breed_classifier.h5"  # Change to .h5 if needed
GOOGLE_DRIVE_FILE_ID = "1B79Kb1IbqeYp7GJeBJm7KVDF1QYQ0xOK"  # Replace with actual Google Drive File ID

# ✅ Download model only if it's missing
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# ✅ Load Model (Ensure it exists)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found. Ensure it is uploaded correctly.")

model = load_model(MODEL_PATH, compile=False)

# ✅ Define Image Size (Should match the training size)
IMAGE_SIZE = (224, 224)

# ✅ FastAPI App
app = FastAPI(title="Dog Breed Classification API", description="Upload an image to predict the dog breed.")

@app.get("/")
def home():
    return {"message": "Welcome to the Dog Breed Classification API!"}


# ✅ Image Preprocessing
def preprocess_image(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model
    return img_array

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return {
            "predicted_breed": predicted_breed,
            "confidence": float(np.max(predictions))
        }

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)