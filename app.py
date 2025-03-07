import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# Désactiver les logs inutiles de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Charger le modèle avec le bon chemin
MODEL_PATH = "vgg_unet_saved_model.keras"  # Le modèle est dans le même dossier que app.py
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "L'API fonctionne correctement !"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    image = file.read()

    # Convertir en tableau numpy
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Format d'image non supporté"}), 400

    # Redimensionner l'image selon les attentes du modèle
    image = cv2.resize(image, (128, 128))  # Modifier selon ton modèle
    image = image / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    prediction = model.predict(image)

    return jsonify({"prediction": prediction.tolist()})  # Convertir en JSON

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port Heroku
    app.run(host='0.0.0.0', port=port)
