import os
from io import BytesIO

# ⚠️ Avant TensorFlow (réduction logs)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image
import boto3

app = Flask(__name__)

# ==============================
# CONFIG S3 / HEROKU
# ==============================
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")  # ex: vgg_unet_trained_model.h5
MODEL_LOCAL_PATH = os.path.join("/tmp", "model.h5")

model = None
MODEL_H = None
MODEL_W = None


def download_model_from_s3():
    if not S3_BUCKET or not S3_MODEL_KEY:
        raise ValueError("Config Vars manquantes : S3_BUCKET et/ou S3_MODEL_KEY (Heroku).")

    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        print(f"✅ Modèle déjà présent : {MODEL_LOCAL_PATH}")
        return

    print(f"📥 Téléchargement du modèle depuis s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_LOCAL_PATH)
    print(f"✅ Modèle téléchargé : {MODEL_LOCAL_PATH} ({os.path.getsize(MODEL_LOCAL_PATH)} bytes)")


def load_model_on_boot():
    global model, MODEL_H, MODEL_W

    download_model_from_s3()

    print("🧠 Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_LOCAL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")

    # Récup taille attendue par le modèle
    # input_shape typique: (None, H, W, 3)
    input_shape = model.input_shape
    MODEL_H = int(input_shape[1])
    MODEL_W = int(input_shape[2])
    print(f"📐 Input attendu par le modèle : H={MODEL_H}, W={MODEL_W}")


# Chargement au démarrage dyno
load_model_on_boot()


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

    # Resize EXACT à la taille attendue par le modèle (ex: 256x512)
    img_resized = cv2.resize(img, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prédiction
    preds = model.predict(img_input, verbose=0)[0]

    # ------------------------------
    # Interprétation sortie
    # ------------------------------
    # Cas multiclasse: (H, W, C) avec C>1
    if preds.ndim == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1).astype("uint8")
        num_classes = preds.shape[-1]
        is_multiclass = True
        out_h, out_w = mask.shape[:2]
    else:
        # Cas binaire: (H, W) ou (H, W, 1)
        if preds.ndim == 3 and preds.shape[-1] == 1:
            preds = preds[:, :, 0]
        # IMPORTANT: masque binaire à la taille exacte de sortie
        mask = (preds > 0.5).astype("uint8")
        num_classes = 2
        is_multiclass = False
        out_h, out_w = mask.shape[:2]

    # ------------------------------
    # Colorisation du masque
    # ------------------------------
    mask_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if is_multiclass:
        palette = [
            (0, 0, 0),        # 0 fond
            (0, 255, 0),      # 1 vert
            (255, 0, 0),      # 2 bleu (BGR)
            (0, 0, 255),      # 3 rouge
            (255, 255, 0),    # 4 cyan
            (255, 0, 255),    # 5 magenta
            (255, 255, 255),  # 6 blanc
        ]
        for cls in range(num_classes):
            mask_rgb[mask == cls] = palette[cls % len(palette)]
    else:
        mask_rgb[mask == 1] = (0, 255, 0)

    # Export PNG
    # (OpenCV = BGR. PIL attend RGB. Ici on crée un masque => ça passe, mais on convertit proprement)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    mask_img = Image.fromarray(mask_rgb)

    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
