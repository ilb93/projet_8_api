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
MODEL_IN_H = None
MODEL_IN_W = None
MODEL_OUT_H = None
MODEL_OUT_W = None
MODEL_OUT_C = None

# Moyennes ImageNet "VGG style" en BGR (exactement ce que ton notebook utilise)
VGG_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


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


def infer_output_shape(m):
    """
    Récupère proprement la shape de sortie.
    Supporte:
      - sortie unique: (None, H, W, C)
      - sortie unique: (None, H, W, 1)
    """
    out_shape = m.output_shape
    # Si multiple outputs, on prend le premier
    if isinstance(out_shape, (list, tuple)) and len(out_shape) > 0 and isinstance(out_shape[0], (list, tuple)):
        out_shape = out_shape[0]

    # typiquement (None, H, W, C)
    h = int(out_shape[1])
    w = int(out_shape[2])
    c = int(out_shape[3]) if len(out_shape) >= 4 else 1
    return h, w, c


def load_model_on_boot():
    global model, MODEL_IN_H, MODEL_IN_W, MODEL_OUT_H, MODEL_OUT_W, MODEL_OUT_C

    download_model_from_s3()

    print("🧠 Chargement du modèle Keras...")
    model = tf.keras.models.load_model(MODEL_LOCAL_PATH, compile=False)
    print("✅ Modèle chargé avec succès !")

    # Input attendu: (None, H, W, 3)
    in_shape = model.input_shape
    MODEL_IN_H = int(in_shape[1])
    MODEL_IN_W = int(in_shape[2])
    print(f"📐 Input attendu par le modèle : H={MODEL_IN_H}, W={MODEL_IN_W}")

    # Output réel du modèle (souvent plus petit)
    MODEL_OUT_H, MODEL_OUT_W, MODEL_OUT_C = infer_output_shape(model)
    print(f"📤 Output du modèle : H={MODEL_OUT_H}, W={MODEL_OUT_W}, C={MODEL_OUT_C}")


# Chargement au démarrage dyno
load_model_on_boot()


def preprocess_like_notebook(img_bgr: np.ndarray) -> np.ndarray:
    """
    Reproduit ton notebook :
    - image BGR (cv2)
    - resize à la taille d'entrée du modèle
    - float32
    - soustraction des moyennes ImageNet (VGG) en BGR
    - batch (1, H, W, 3)
    """
    img_resized = cv2.resize(img_bgr, (MODEL_IN_W, MODEL_IN_H), interpolation=cv2.INTER_AREA)
    x = img_resized.astype(np.float32)
    x -= VGG_MEAN_BGR  # BGR mean subtraction (IMPORTANT)
    x = np.expand_dims(x, axis=0)
    return x


def decode_image_from_request(file_storage):
    img_bytes = file_storage.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # BGR
    return img_bgr


def build_palette(num_classes: int):
    """
    Palette simple mais robuste : si num_classes > palette, on boucle.
    Les couleurs sont en BGR (car on remplit via OpenCV-style puis on convertit vers RGB à la fin).
    """
    base = [
        (0, 0, 0),        # 0 background
        (0, 255, 0),      # 1 green
        (255, 0, 0),      # 2 blue
        (0, 0, 255),      # 3 red
        (255, 255, 0),    # 4 cyan
        (255, 0, 255),    # 5 magenta
        (0, 255, 255),    # 6 yellow
        (255, 255, 255),  # 7 white
        (80, 80, 80),     # 8 gray
        (0, 128, 255),    # 9 orange-ish
    ]
    return [base[i % len(base)] for i in range(num_classes)]


def postprocess_prediction(preds: np.ndarray):
    """
    Convertit la sortie modèle en mask uint8 (H_out, W_out).
    - Multiclasse: argmax sur axis=-1
    - Binaire: threshold 0.5
    """
    # preds typique : (H_out, W_out, C) ou (H_out, W_out, 1) ou (H_out, W_out)
    if preds.ndim == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1).astype(np.uint8)
        num_classes = int(preds.shape[-1])
        return mask, num_classes, True

    # binaire
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds[:, :, 0]
    mask = (preds > 0.5).astype(np.uint8)
    return mask, 2, False


def colorize_mask(mask: np.ndarray, num_classes: int, is_multiclass: bool) -> np.ndarray:
    """
    Retourne une image BGR uint8 colorisée de shape (H, W, 3)
    """
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if is_multiclass:
        palette = build_palette(num_classes)
        for cls in range(num_classes):
            out[mask == cls] = palette[cls]
    else:
        out[mask == 1] = (0, 255, 0)  # green for foreground

    return out


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation opérationnelle"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["file"]
    img_bgr = decode_image_from_request(file)

    if img_bgr is None:
        return jsonify({"error": "Format d'image invalide"}), 400

    # Taille originale (pour upsample final en nearest)
    orig_h, orig_w = img_bgr.shape[:2]

    # ✅ Preprocess exactement comme notebook
    x = preprocess_like_notebook(img_bgr)

    # Prédiction
    preds = model.predict(x, verbose=0)

    # Certains modèles renvoient une liste
    if isinstance(preds, list):
        preds = preds[0]

    # On enlève batch
    preds = preds[0]

    # ✅ Mask classes (argmax/threshold)
    mask, num_classes, is_multiclass = postprocess_prediction(preds)

    # ✅ Colorisation à la taille de sortie réelle
    mask_bgr = colorize_mask(mask, num_classes, is_multiclass)

    # ✅ Upsample (NEAREST) vers la taille originale pour affichage
    mask_bgr_up = cv2.resize(mask_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Conversion RGB pour PIL
    mask_rgb_up = cv2.cvtColor(mask_bgr_up, cv2.COLOR_BGR2RGB)
    mask_img = Image.fromarray(mask_rgb_up)

    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
