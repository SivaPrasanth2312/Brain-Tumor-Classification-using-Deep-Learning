import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request

app = Flask(__name__)

model = tf.keras.models.load_model("model/brain_tumor_cnn.h5")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "⚠ Tumor Detected", pred
    else:
        return "✅ No Tumor Detected", pred

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result, confidence = predict_image(path)
            image_path = path

    return render_template("index.html", result=result, confidence=confidence, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)