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

# On utilise maintenant le modèle .h5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "unet_final_model.h5")

# Réduire les logs TensorFlow
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

    # ==============================
    # PRÉDICTION DU MODÈLE
    # ==============================
    preds = model.predict(img_input)[0]

    # ==============================
    # BINAIRE OU MULTICLASSE
    # ==============================
    # Cas multiclasse : shape (H, W, C) avec C > 1 → argmax
    if preds.ndim == 3 and preds.shape[-1] > 1:
        mask = np.argmax(preds, axis=-1).astype("uint8")  # valeurs 0..C-1
        num_classes = preds.shape[-1]
        is_multiclass = True
    else:
        # Cas binaire : probas → threshold 0.5
        if preds.ndim == 3:
            preds = preds[:, :, 0]
        mask = (preds > 0.5).astype("uint8")  # 0 ou 1
        num_classes = 2
        is_multiclass = False

    # ==============================
    # COLORISATION DU MASQUE
    # ==============================

    # Image RGB vide
    mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    if is_multiclass:
        # Petite palette de couleurs (tu peux changer si tu veux d'autres couleurs)
        palette = [
            (0, 0, 0),        # classe 0 = fond = noir
            (0, 255, 0),      # classe 1 = vert
            (255, 0, 0),      # classe 2 = bleu
            (0, 0, 255),      # classe 3 = rouge
            (255, 255, 0),    # classe 4 = cyan
            (255, 0, 255),    # classe 5 = magenta
            (255, 255, 255),  # classe 6 = blanc
        ]

        # On boucle sur chaque classe présente
        for cls in range(num_classes):
            color = palette[cls % len(palette)]
            mask_rgb[mask == cls] = color

    else:
        # Binaire : fond noir, objet en vert bien visible
        mask_rgb[mask == 1] = (0, 255, 0)

    # ==============================
    # EXPORT PNG
    # ==============================

    mask_img = Image.fromarray(mask_rgb)
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# ==============================
# MODE LOCAL
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # En local tu peux laisser debug=True, sur Heroku il s'en fiche
    app.run(host="0.0.0.0", port=port, debug=True)
