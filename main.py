import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN for compatibility with TensorFlow 2.6.0

# ✅ Load the trained model
MODEL_PATH = "model.keras"  # Update this if using .h5

model = load_model(MODEL_PATH)

# ✅ Define image size and class labels (update as per your dataset)
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ["Breed1", "Breed2", "Breed3", "Breed4", "Breed5", "Breed6", "Breed7", "Breed8"]  # Update with actual breed names

app = FastAPI(title="Dog Breed Classification API", description="Upload an image to predict the dog breed.")

# ✅ Preprocess uploaded image
def preprocess_image(file: UploadFile):
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_breed = CLASS_LABELS[predicted_class]

    return {"predicted_breed": predicted_breed, "confidence": float(np.max(predictions))}

# ✅ Run FastAPI Server (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
