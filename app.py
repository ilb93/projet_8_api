import os
from io import BytesIO

# ⚠️ Mettre les variables AVANT d'importer tensorflow pour réduire les logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image

import boto3


# ==============================
# CONFIG GÉNÉRALE
# ==============================

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Variables Heroku (Config Vars)
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")  # ex: vgg_unet_trained_model.h5

# Chemin local Heroku (disque éphémère OK)
MODEL_LOCAL_PATH = os.path.join("/tmp", "model.h5")

app = Flask(__name__)

model = None  # chargé au démarrage


# ==============================
# S3 -> DOWNLOAD MODEL
# ==============================

def download_model_from_s3():
    if not S3_BUCKET or not S3_MODEL_KEY:
        raise ValueError(
            "Config Vars manquantes. Vérifie S3_BUCKET et S3_MODEL_KEY dans Heroku."
        )

    # Si déjà téléchargé, on ne retélécharge pas
    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        print(f"✅ Modèle déjà présent : {MODEL_LOCAL_PATH}")
        return

    print(f"📥 Téléchargement du modèle depuis s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_LOCAL_PATH)
    print(f"✅ Modèle téléchargé : {MODEL_LOCAL_PATH} ({os.path.getsize(MODEL_LOCAL_PATH)} bytes)")


def load_model():
    global model
    download_model_from_s3()
    print("🧠 Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_LOCAL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")


# Charger au démarrage (Heroku boot)
load_model()


# ==============================
# ROUTES API
# ==============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation opérationnelle"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    # Décodage OpenCV
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Format d'image invalide"}), 400

    # Resize au format du modèle
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # ==============================
    # PRÉDICTION DU MODÈLE
    # ==============================
    preds = model.predict(img_input)[0]

    # ==============================
    # BINAIRE OU MULTICLASSE
    # ==============================
    if preds.ndim == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1).astype("uint8")  # 0..C-1
        num_classes = preds.shape[-1]
        is_multiclass = True
    else:
        if preds.ndim == 3:
            preds = preds[:, :, 0]
        mask = (preds > 0.5).astype("uint8")
        num_classes = 2
        is_multiclass = False

    # ==============================
    # COLORISATION DU MASQUE
    # ==============================
    mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    if is_multiclass:
        palette = [
            (0, 0, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (255, 255, 255),
        ]
        for cls in range(num_classes):
            color = palette[cls % len(palette)]
            mask_rgb[mask == cls] = color
    else:
        mask_rgb[mask == 1] = (0, 255, 0)

    # ==============================
    # EXPORT PNG
    # ==============================
    mask_img = Image.fromarray(mask_rgb)
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
