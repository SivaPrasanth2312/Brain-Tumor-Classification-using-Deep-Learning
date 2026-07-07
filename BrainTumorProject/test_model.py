import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 128

# Load trained model
model = load_model("brain_tumor_model.h5")

# Image path (change if needed)
img_path = "dataset/yes/Y1.jpg"

# Load image
img = Image.open(img_path).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output
if prediction >= 0.5:
    print(f"🔴 Tumor Detected! (Confidence: {prediction*100:.1f}%)")
else:
    print(f"🟢 No Tumor Found. (Confidence: {(1-prediction)*100:.1f}%)")