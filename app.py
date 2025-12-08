import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from io import BytesIO

# 📌 Initialisation de l'application Flask
app = Flask(__name__)

# 📌 Chargement du modèle
MODEL_PATH = "model_p8.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Erreur : Le modèle {MODEL_PATH} est introuvable.")
model = tf.keras.models.load_model(MODEL_PATH)

# 📌 Définition du nombre de classes et de la palette de couleurs
N_CLASSES = 8
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Fond (noir)
    [255, 0, 0],      # Classe 1 (rouge)
    [0, 255, 0],      # Classe 2 (vert)
    [0, 0, 255],      # Classe 3 (bleu)
    [255, 255, 0],    # Classe 4 (jaune)
    [255, 0, 255],    # Classe 5 (magenta)
    [0, 255, 255],    # Classe 6 (cyan)
    [255, 255, 255]   # Classe 7 (blanc)
], dtype=np.uint8)

# 📌 Fonction de prétraitement des images
def preprocess_image(image):
    """Convertit l'image en un format compatible avec le modèle"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB si nécessaire

    # 📌 Vérification de la forme de l'image et correction
    if image.shape[:2] == (512, 256):  
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Pivoter si l'image est inversée

    image = cv2.resize(image, (512, 256))  # Redimensionner correctement
    image = image.astype(np.float32) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension batch
    return image

# 📌 Fonction d'affichage des prédictions
def apply_color_map(mask):
    """Applique un masque de couleurs sur la segmentation prédite."""
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]  # Convertir (H, W, 1) en (H, W)

    h, w = mask.shape  # Vérification
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(N_CLASSES):
        if np.any(mask == class_idx):  # Vérification si la classe est bien présente
            color_mask[mask == class_idx] = COLOR_PALETTE[class_idx]

    return color_mask

# 📌 Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📌 Récupération du fichier
        if 'file' not in request.files:
            return jsonify({"error": "❌ Aucun fichier reçu"}), 400
        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "❌ Erreur lors de la lecture de l'image"}), 400

        # 📌 Prétraitement
        image_input = preprocess_image(image)

        # 📌 Prédiction
        prediction = model.predict(image_input)

        # 📌 Vérification de la forme de sortie du modèle
        if prediction.shape[-1] != N_CLASSES:
            return jsonify({"error": f"❌ Erreur : Nombre de classes inattendu en sortie: {prediction.shape[-1]}"}), 500

        # 📌 Génération du masque (assure qu'il est en 2D)
        class_map = np.argmax(prediction, axis=-1)

        # 📌 Correction pour s'assurer que class_map est bien (H, W)
        if len(class_map.shape) == 3:
            class_map = np.squeeze(class_map, axis=0)

        print(f"🧐 Valeurs uniques dans le masque prédictif: {np.unique(class_map)}")

        # 🔍 **Débogage : Sauvegarde temporaire du masque brut**
        cv2.imwrite("mask_raw.png", (class_map * 30).astype(np.uint8))  # Sauvegarde pour vérif

        # 📌 Application du masque de couleur
        colorized_mask = apply_color_map(class_map)

        # 🔍 **Débogage : Sauvegarde du masque colorisé**
        cv2.imwrite("mask_colorized.png", colorized_mask)  # Vérif que la couleur est appliquée

        # 📌 Convertir l'image en réponse
        _, img_encoded = cv2.imencode('.png', colorized_mask)
        img_bytes = BytesIO(img_encoded.tobytes())

        return send_file(img_bytes, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"❌ Erreur interne : {str(e)}"}), 500

# 📌 Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
