import os
from io import BytesIO

import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from PIL import Image

# ==============================
# CONFIG GÉNÉRALE
# ==============================

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Le modèle est LOCAL dans ton repo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vgg_unet_saved_model.keras")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle introuvable : {MODEL_PATH}")

print(f"📥 Chargement du modèle depuis : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé avec succès !")


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

    # Prédiction
    preds = model.predict(img_input)[0]

    # ========================
    # BILAIRE ou MULTICLASSE
    # ========================
    if preds.ndim == 3 and preds.shape[-1] > 1:
        # multi-classes → on prend argmax
        mask = np.argmax(preds, axis=-1).astype("uint8")
    else:
        # binaire
        if preds.ndim == 3:
            preds = preds[:, :, 0]
        mask = (preds > 0.5).astype("uint8")

    # ========================
    # EXPORT EN NIVEAUX DE GRIS
    # ========================
    mask_img = Image.fromarray((mask * 255).astype("uint8"))
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# ==============================
# MODE LOCAL
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
