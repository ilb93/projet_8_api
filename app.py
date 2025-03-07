from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import io

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model("model/vgg_unet_saved_model.keras")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "L'API fonctionne correctement !"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    image = file.read()

    # Prétraiter l'image
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))  # Ajuster la taille selon ton modèle
    image = image / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return jsonify({"prediction": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
