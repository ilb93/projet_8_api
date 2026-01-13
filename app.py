import os
from io import BytesIO

# avant TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image
import boto3
import random

app = Flask(__name__)

# ==============================
# CONFIG S3 / HEROKU
# ==============================
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")
MODEL_LOCAL_PATH = os.path.join("/tmp", "model.h5")

model = None

# tailles modèle (comme notebook)
MODEL_IN_H = None
MODEL_IN_W = None
MODEL_OUT_H = None
MODEL_OUT_W = None
N_CLASSES = 8  # ton notebook = 8 classes remappées

# =========================================================
# Palette EXACTEMENT comme ton notebook (seed 24)
# generate_colors(n_classes) -> couleurs RGB normalisées
# ici on génère des RGB 0..255 directement
# =========================================================
def generate_colors_uint8(n_classes: int):
    random.seed(24)
    colors = [(random.randint(0, 255),
               random.randint(0, 255),
               random.randint(0, 255)) for _ in range(n_classes)]
    return colors  # RGB uint8

PALETTE_RGB = generate_colors_uint8(N_CLASSES)


def download_model_from_s3():
    if not S3_BUCKET or not S3_MODEL_KEY:
        raise ValueError("Config Vars manquantes : S3_BUCKET et/ou S3_MODEL_KEY.")

    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        print(f"✅ Modèle déjà présent : {MODEL_LOCAL_PATH}")
        return

    print(f"📥 Téléchargement du modèle depuis s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_LOCAL_PATH)
    print(f"✅ Modèle téléchargé : {MODEL_LOCAL_PATH} ({os.path.getsize(MODEL_LOCAL_PATH)} bytes)")


def load_model_on_boot():
    global model, MODEL_IN_H, MODEL_IN_W, MODEL_OUT_H, MODEL_OUT_W

    download_model_from_s3()

    print("🧠 Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_LOCAL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")

    # Input shape: (None, H, W, 3)
    in_shape = model.input_shape
    MODEL_IN_H = int(in_shape[1])
    MODEL_IN_W = int(in_shape[2])

    # Output shape peut être (None, H, W, C) OU aplati.
    # On récupère H/W si possible, sinon fallback sur variables custom si elles existent.
    out_shape = model.output_shape
    if isinstance(out_shape, (list, tuple)) and len(out_shape) > 0 and isinstance(out_shape[0], (list, tuple)):
        out_shape = out_shape[0]

    # Cas standard: (None, H, W, C)
    if isinstance(out_shape, (list, tuple)) and len(out_shape) == 4:
        MODEL_OUT_H = int(out_shape[1])
        MODEL_OUT_W = int(out_shape[2])
    else:
        # fallback: on déduit par ratio (souvent /2)
        # ton notebook indique output (128,256) quand input (256,512)
        MODEL_OUT_H = MODEL_IN_H // 2
        MODEL_OUT_W = MODEL_IN_W // 2

    print(f"📐 Input attendu : H={MODEL_IN_H}, W={MODEL_IN_W}")
    print(f"📤 Output attendu : H={MODEL_OUT_H}, W={MODEL_OUT_W}, C={N_CLASSES}")


load_model_on_boot()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation opérationnelle"})


def decode_image(file_storage):
    img_bytes = file_storage.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img_bgr


def preprocess_like_notebook(img_bgr: np.ndarray) -> np.ndarray:
    """
    EXACT ton notebook:
      im2p = cv2.resize(im2p, (model.input_width, model.input_height)).astype(np.float32)
      im2p = np.expand_dims(im2p, axis=0)
    """
    img_resized = cv2.resize(img_bgr, (MODEL_IN_W, MODEL_IN_H), interpolation=cv2.INTER_AREA)
    x = img_resized.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def reshape_prediction_like_notebook(preds: np.ndarray) -> np.ndarray:
    """
    EXACT ton notebook:
      im_p = model.predict(im2p)
      im_p = im_p.reshape((model.output_height, model.output_width, len(class_names))).argmax(axis=2)

    On gère 3 cas:
    - (H,W,C) -> ok
    - (H*W,C) -> reshape (H,W,C)
    - (H*W*C,) -> reshape (H,W,C)
    """
    # enlever batch si présent
    if preds.ndim >= 1 and preds.shape[0] == 1:
        preds = preds[0]

    # déjà (H,W,C)
    if preds.ndim == 3:
        hwc = preds

    # (H*W, C)
    elif preds.ndim == 2:
        c = preds.shape[-1]
        hwc = preds.reshape((MODEL_OUT_H, MODEL_OUT_W, c))

    # (H*W*C,)
    elif preds.ndim == 1:
        hwc = preds.reshape((MODEL_OUT_H, MODEL_OUT_W, N_CLASSES))

    else:
        raise ValueError(f"Shape sortie non gérée: {preds.shape}")

    # argmax classes
    mask = hwc.argmax(axis=2).astype(np.uint8)
    return mask


def colorize_mask_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Mask (H,W) -> image RGB uint8 avec palette seed(24)
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(N_CLASSES):
        out[mask == cls] = PALETTE_RGB[cls]
    return out


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["file"]
    img_bgr = decode_image(file)
    if img_bgr is None:
        return jsonify({"error": "Format d'image invalide"}), 400

    orig_h, orig_w = img_bgr.shape[:2]

    # ✅ preprocess EXACT notebook
    x = preprocess_like_notebook(img_bgr)

    # predict
    preds = model.predict(x, verbose=0)
    if isinstance(preds, list):
        preds = preds[0]

    # ✅ reshape + argmax EXACT notebook
    try:
        mask = reshape_prediction_like_notebook(preds)
    except Exception as e:
        return jsonify({"error": f"Postprocess failed: {str(e)}"}), 500

    # colorize RGB
    mask_rgb = colorize_mask_rgb(mask)

    # ✅ upsample NEAREST à taille originale (comme ton notebook pour display)
    mask_rgb_up = cv2.resize(mask_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # export PNG
    img_pil = Image.fromarray(mask_rgb_up)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

import botocore
from flask import Response

S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", S3_BUCKET)  # même bucket si tu veux
S3_LEFT_PREFIX = os.getenv("S3_LEFT_PREFIX", "leftImg8bit/")
S3_GT_PREFIX   = os.getenv("S3_GT_PREFIX", "gtfine/")

s3_data = boto3.client("s3", region_name=AWS_REGION)

def s3_read(key: str) -> bytes:
    obj = s3_data.get_object(Bucket=S3_DATA_BUCKET, Key=key)
    return obj["Body"].read()


@app.route("/ids", methods=["GET"])
def ids():
    """
    Retourne la liste des IDs disponibles (base commune image/gt).
    ID format: aachen_000010_000019
    """
    paginator = s3_data.get_paginator("list_objects_v2")
    ids_list = []

    for page in paginator.paginate(Bucket=S3_DATA_BUCKET, Prefix=S3_LEFT_PREFIX):
        for it in page.get("Contents", []):
            k = it["Key"]
            if not k.lower().endswith(".png"):
                continue
            filename = k.split("/")[-1]  # ex aachen_000010_000019_leftImg8bit.png
            if filename.endswith("_leftImg8bit.png"):
                img_id = filename.replace("_leftImg8bit.png", "")
                ids_list.append(img_id)

    ids_list = sorted(list(set(ids_list)))
    return jsonify({"ids": ids_list})


@app.route("/image/<img_id>", methods=["GET"])
def get_image(img_id):
    # on cherche dans aachen/ (car tu as organisé par ville)
    # img_id commence déjà par aachen_... donc on sait la ville
    city = img_id.split("_")[0]
    key = f"{S3_LEFT_PREFIX}{city}/{img_id}_leftImg8bit.png"
    try:
        data = s3_read(key)
        return Response(data, mimetype="image/png")
    except botocore.exceptions.ClientError:
        return jsonify({"error": "Image introuvable"}), 404


@app.route("/gt/<img_id>", methods=["GET"])
def get_gt(img_id):
    city = img_id.split("_")[0]
    key = f"{S3_GT_PREFIX}{city}/{img_id}_gtFine_labelIds.png"
    try:
        data = s3_read(key)
        return Response(data, mimetype="image/png")
    except botocore.exceptions.ClientError:
        return jsonify({"error": "GT introuvable"}), 404


@app.route("/predict_by_id/<img_id>", methods=["GET"])
def predict_by_id(img_id):
    city = img_id.split("_")[0]
    key = f"{S3_LEFT_PREFIX}{city}/{img_id}_leftImg8bit.png"

    try:
        img_bytes = s3_read(key)
    except botocore.exceptions.ClientError:
        return jsonify({"error": "Image introuvable"}), 404

    # decode OpenCV
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Image invalide"}), 400

    # réutilise ton pipeline actuel (celui qui marche)
    orig_h, orig_w = img.shape[:2]

    x = preprocess_like_notebook(img)
    preds = model.predict(x, verbose=0)
    if isinstance(preds, list):
        preds = preds[0]

    mask = reshape_prediction_like_notebook(preds)
    mask_rgb = colorize_mask_rgb(mask)
    mask_rgb_up = cv2.resize(mask_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    out_pil = Image.fromarray(mask_rgb_up)
    buf = BytesIO()
    out_pil.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
