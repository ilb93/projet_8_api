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

# Taille d’entrée du modèle
IMG_HEIGHT = 256
IMG_WIDTH = 512  # VGG-UNet entraîné en 256x512

# ID du fichier Google Drive (modèle .h5)
DRIVE_FILE_ID = "1k3wtDmMviqrysyw1dwzpJIUQ5J0Of_yR"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Fichier modèle *local* (dans le dyno Heroku)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_p8.h5")

# Palette Cityscapes simplifiée
CITYSCAPES_COLORMAP = {
    0: (0, 0, 0),          # fond
    1: (128, 64, 128),     # route
    2: (244, 35, 232),     # trottoir
    3: (70, 70, 70),       # bâtiments
    4: (107, 142, 35),     # végétation
    5: (70, 130, 180),     # ciel
    6: (220, 20, 60),      # piétons
    7: (255, 0, 0),        # voitures
}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)


# ==============================
# TÉLÉCHARGEMENT DU MODÈLE
# ==============================

def download_model_if_needed():
    """
    Télécharge le modèle depuis Google Drive si le fichier local n'existe pas.
    Utilise gdown qui gère les gros fichiers et les confirmations Drive.
    """
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modèle déjà présent : {MODEL_PATH}")
        return

    print("⬇️  Téléchargement du modèle depuis Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Téléchargement du modèle échoué, fichier introuvable.")


# ==============================
# CHARGEMENT DU MODÈLE
# ==============================

print("🚀 Initialisation de l'API...")

try:
    download_model_if_needed()
    print(f"✅ Chargement du modèle depuis : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print("❌ Erreur lors du téléchargement/chargement du modèle :", e)
    model = None  # Optionnel : pour éviter que l'app crashe complètement


# ==============================
# UTILS
# ==============================

def apply_colormap(mask: np.ndarray) -> np.ndarray:
    """
    mask : (H, W) avec des entiers de classes
    -> retourne une image couleur (H, W, 3) en uint8
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in CITYSCAPES_COLORMAP.items():
        color_mask[mask == label] = color

    return color_mask


# ==============================
# ROUTES FLASK
# ==============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation projet 8 opérationnelle"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé sur le serveur"}), 500

    # Vérification de l'image
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée (clé 'file' manquante)"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Décodage OpenCV
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Format d'image non supporté"}), 400

    # Redimensionnement au format du modèle
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # (W, H) pour OpenCV

    # Normalisation + batch
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Prédiction
    try:
        preds = model.predict(image)[0]  # (H, W, C) ou (H, W, 1)
    except Exception as e:
        return jsonify({"error": f"Erreur interne dans le modèle : {str(e)}"}), 500

    # Multi-classes vs binaire
    if len(preds.shape) == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1).astype("int32")  # (H, W)
    else:
        mask = preds
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype("int32")

    # Colorisation
    color_mask = apply_colormap(mask)

    # Conversion en PNG
    pil_mask = Image.fromarray(color_mask)
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
