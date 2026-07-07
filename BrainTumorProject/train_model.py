import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128
DATASET_PATH = "dataset"

def load_data():
    images = []
    labels = []

    categories = {"yes": 1, "no": 0}

    for category, label in categories.items():
        folder = os.path.join(DATASET_PATH, category)

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0

                images.append(img_array)
                labels.append(label)
            except:
                pass

    return np.array(images), np.array(labels)

print("Loading dataset...")
X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3)

model.save("brain_tumor_model.h5")

print("✅ DONE - Model saved")