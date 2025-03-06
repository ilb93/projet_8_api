from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# 🔹 Initialisation de l'application Flask
app = Flask(__name__)

# 🔹 Chargement du modèle
MODEL_PATH = "model/vgg_unet_saved_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# 🔹 Définition des classes
CLASS_NAMES = ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]

# 🔹 Fonction de prédiction
def predict_image(image):
    img = cv2.resize(image, (model.input_shape[1], model.input_shape[2]))  # Redimensionner l'image
    img = img.astype(np.float32) / 255.0  # Normalisation
    img = np.expand_dims(img, axis=0)  # Ajouter la dimension batch

    prediction = model.predict(img)  # Prédiction
    pred_class = np.argmax(prediction, axis=-1)[0]  # Classe prédite

    return pred_class.tolist()

# 🔹 Route API pour la prédiction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    pred = predict_image(image)
    return jsonify({"prediction": pred})

# 🔹 Lancer l'API
if __name__ == "__main__":
    app.run(debug=True)
