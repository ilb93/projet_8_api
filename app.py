import os
from io import BytesIO

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image
import gdown


# ==============================
# CONFIG GÉNÉRALE
# ==============================

IMG_HEIGHT = 256
IMG_WIDTH = 512  # Format attendu par ton modèle

# ID DU FICHIER GOOGLE DRIVE (100% correct maintenant)
MODEL_DRIVE_ID = "1k3wtDmMviqrysyw1dwzpJIUQ5J0Of_yR"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

# Chemin local (Heroku)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_p8.h5")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)


# ==============================
# TÉLÉCHARGEMENT DU MODÈLE
# ==============================

def download_model_if_needed():
    """Télécharge le modèle depuis Google Drive si absent."""
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modèle déjà présent : {MODEL_PATH}")
        return

    print("⬇️  Téléchargement du modèle depuis Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé avec succès !")
    except Exception as e:
        print("❌ Erreur téléchargement modèle :", e)
        raise e


# ==============================
# FONCTION SANS COULEURS (grayscale)
# ==============================

def apply_colormap(mask: np.ndarray) -> np.ndarray:
    """Retourne le masque brut en niveaux de gris (comme au tout début)."""

    max_val = mask.max() if mask.max() > 0 else 1
    mask_norm = (mask.astype("float32") / max_val) * 255.0
    return mask_norm.astype("uint8")


# ==============================
# CHARGEMENT DU MODÈLE
# ==============================

print("🚀 Initialisation API…")
download_model_if_needed()

print(f"📦 Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé !")


# ==============================
# ROUTES API
# ==============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API opérationnelle"})


@app.route("/predict", methods=["POST"])
def predict():
    # Vérification
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Décodage OpenCV
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Image invalide"}), 400

    # Resize au format du modèle
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalisation + batch
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, 0)

    # Prédiction
    try:
        preds = model.predict(image)[0]
    except Exception as e:
        return jsonify({"error": f"Erreur modèle : {str(e)}"}), 500

    # Traitement du masque selon format
    if len(preds.shape) == 3 and preds.shape[-1] > 1:
        # Multi-classes → argmax
        mask = np.argmax(preds, axis=-1)
    else:
        # Binaire
        mask = preds
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype("uint8")

    # Applique un masque en niveaux de gris
    gray_mask = apply_colormap(mask)

    # Envoi PNG
    pil_mask = Image.fromarray(gray_mask)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# ==============================
# LANCEMENT LOCAL
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
