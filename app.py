import os
from io import BytesIO

# Mettre AVANT tensorflow (réduit les logs)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image
import boto3

# ==============================
# CONFIG AWS / S3 (Heroku Config Vars)
# ==============================
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")  # ex: vgg_unet_trained_model.h5

MODEL_LOCAL_PATH = "/tmp/model.h5"

app = Flask(__name__)

model = None
INPUT_H = INPUT_W = None
OUT_H = OUT_W = None
OUT_C = None


def download_model_from_s3():
    if not S3_BUCKET or not S3_MODEL_KEY:
        raise ValueError("Config Vars manquantes : S3_BUCKET et/ou S3_MODEL_KEY")

    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        print(f"✅ Modèle déjà présent : {MODEL_LOCAL_PATH}")
        return

    print(f"📥 Téléchargement du modèle depuis s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_LOCAL_PATH)
    print(f"✅ Modèle téléchargé : {MODEL_LOCAL_PATH} ({os.path.getsize(MODEL_LOCAL_PATH)} bytes)")


def infer_shapes_from_model(m):
    """Déduit input/output du modèle pour éviter les erreurs (256x512, sorties aplaties, etc.)."""
    global INPUT_H, INPUT_W, OUT_H, OUT_W, OUT_C

    # input_shape: (None, H, W, 3) ou parfois une liste
    in_shape = m.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    INPUT_H, INPUT_W = int(in_shape[1]), int(in_shape[2])

    out_shape = m.output_shape
    if isinstance(out_shape, list):
        out_shape = out_shape[0]

    # Cas classique segmentation: (None, H, W, C)
    if len(out_shape) == 4:
        OUT_H, OUT_W, OUT_C = int(out_shape[1]), int(out_shape[2]), int(out_shape[3])
    # Cas sortie aplatie: (None, N) ou (None, N, C)
    elif len(out_shape) == 3:
        # (None, N, C)
        n = int(out_shape[1])
        OUT_C = int(out_shape[2])
        # On essaye de retrouver (H, W) via l’input (souvent identique)
        # Si ça ne colle pas, on forcera un reshape via n plus bas.
        OUT_H, OUT_W = INPUT_H, INPUT_W
        if OUT_H * OUT_W != n:
            # fallback : on essaye de déduire un W plausible
            # (ça couvre ton cas 32768)
            if n % INPUT_H == 0:
                OUT_H, OUT_W = INPUT_H, n // INPUT_H
            elif n % INPUT_W == 0:
                OUT_W, OUT_H = INPUT_W, n // INPUT_W
            else:
                # dernier recours
                OUT_H, OUT_W = 1, n
    elif len(out_shape) == 2:
        # (None, N)
        n = int(out_shape[1])
        OUT_C = 1
        OUT_H, OUT_W = INPUT_H, INPUT_W
        if OUT_H * OUT_W != n:
            if n % INPUT_H == 0:
                OUT_H, OUT_W = INPUT_H, n // INPUT_H
            elif n % INPUT_W == 0:
                OUT_W, OUT_H = INPUT_W, n // INPUT_W
            else:
                OUT_H, OUT_W = 1, n
    else:
        raise ValueError(f"Shape de sortie non gérée : {out_shape}")

    print(f"📐 Model input : {INPUT_H}x{INPUT_W} | output : {OUT_H}x{OUT_W} | C={OUT_C}")


def load_model():
    global model
    download_model_from_s3()
    print("🧠 Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_LOCAL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")
    infer_shapes_from_model(model)


# Chargement au boot Heroku
load_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation opérationnelle"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    img_bytes = request.files["file"].read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Format d'image invalide"}), 400

    # Resize selon le modèle (pas en dur)
    img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    preds = model.predict(img_input, verbose=0)[0]
    print("🔎 preds.shape =", getattr(preds, "shape", None))

    # ====== Remettre preds en (H, W, C) si c’est aplati ======
    if preds.ndim == 1:
        # (N,) -> (H,W)
        preds = preds.reshape((OUT_H, OUT_W))
    elif preds.ndim == 2:
        # (N,C) -> (H,W,C)
        preds = preds.reshape((OUT_H, OUT_W, preds.shape[-1]))

    # ====== Binaire vs Multiclasse ======
    is_multiclass = (preds.ndim == 3 and preds.shape[-1] > 1)

    if is_multiclass:
        mask = np.argmax(preds, axis=-1).astype("uint8")
        num_classes = preds.shape[-1]
    else:
        if preds.ndim == 3:
            preds = preds[:, :, 0]
        mask = (preds > 0.5).astype("uint8")
        num_classes = 2

    # ====== Colorisation ======
    mask_rgb = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

    if is_multiclass:
        palette = [
            (0, 0, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (255, 255, 255),
            (0, 255, 255),
        ]
        for cls in range(num_classes):
            mask_rgb[mask == cls] = palette[cls % len(palette)]
    else:
        mask_rgb[mask == 1] = (0, 255, 0)

    # Export PNG
    mask_img = Image.fromarray(mask_rgb)
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
