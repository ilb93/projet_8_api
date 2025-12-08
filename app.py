import os
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image

# =======================================================================
# CONFIG
# =======================================================================

# Taille d'entrée attendue par le modèle (vgg_unet_256_512)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Chemin du modèle (adapter l'extension si besoin : .h5 ou .keras)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_p8.h5")

# Désactiver les logs verbeux de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# =======================================================================
# CHARGEMENT DU MODELE
# =======================================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle introuvable : {MODEL_PATH}")

print(f"✅ Chargement du modèle depuis : {MODEL_PATH}")
# compile=False pour éviter les problèmes de custom loss (combined_loss, etc.)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé avec succès !")


# =======================================================================
# ROUTE DE TEST
# =======================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation projet 8 opérationnelle"})


# =======================================================================
# ROUTE DE PREDICTION
# =======================================================================

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée (clé 'file' manquante)"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Décodage de l'image
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Format d'image non supporté"}), 400

    # ⚠️ Redimensionner dans le format attendu par le modèle
    # OpenCV : (largeur, hauteur)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalisation + ajout de la dimension batch
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Prédiction
    try:
        preds = model.predict(image)[0]  # (H, W, C) ou (H, W, 1)
    except Exception as e:
        return jsonify({"error": f"Erreur interne dans le modèle : {str(e)}"}), 500

    # Cas le plus fréquent : sortie (H, W, nb_classes) -> on prend l'argmax
    if len(preds.shape) == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1)  # (H, W)
    else:
        # Cas binaire : (H, W, 1) ou (H, W) -> seuillage à 0.5
        mask = preds
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype("uint8")  # 0 ou 1

    # Mise à l'échelle sur 0-255 pour en faire une image
    mask_uint8 = (mask.astype("float32") / max(mask.max(), 1) * 255).astype("uint8")

    # Conversion en image PNG
    pil_mask = Image.fromarray(mask_uint8)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)

    # On renvoie directement l'image PNG
    return send_file(buf, mimetype="image/png")


# =======================================================================
# LANCEMENT LOCAL (Heroku utilisera gunicorn via le Procfile)
# =======================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
