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

# Taille d'entrée attendue par le modèle : 256 x 256
IMG_HEIGHT = 256
IMG_WIDTH = 512

# Nom du fichier modèle dans ton repo
#   -> adapte l'extension si besoin : "model_p8.h5" ou "model_p8.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_p8.keras")

# Désactiver les logs verbeux de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# Palette Cityscapes simplifiée (8 classes principales)
CITYSCAPES_COLORMAP = {
    0: (0, 0, 0),          # background - noir
    1: (128, 64, 128),     # route - violet
    2: (244, 35, 232),     # trottoir - rose
    3: (70, 70, 70),       # bâtiment - gris foncé
    4: (107, 142, 35),     # végétation - vert
    5: (70, 130, 180),     # ciel - bleu
    6: (220, 20, 60),      # piétons - rouge
    7: (255, 0, 0),        # voitures - rouge vif
}

# =======================================================================
# CHARGEMENT DU MODELE
# =======================================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modèle introuvable : {MODEL_PATH}")

print(f"✅ Chargement du modèle depuis : {MODEL_PATH}")
# compile=False pour éviter les problèmes de custom loss
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé avec succès !")


# =======================================================================
# UTILS
# =======================================================================

def apply_colormap(mask: np.ndarray) -> np.ndarray:
    """
    mask : tableau (H, W) avec des entiers de classe (0, 1, 2, ...)
    retourne : image couleur (H, W, 3) en uint8
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in CITYSCAPES_COLORMAP.items():
        color_mask[mask == label] = color

    # Tous les labels non définis dans la colormap restent noirs (0,0,0)
    return color_mask


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

    # Redimensionnement dans le format attendu par le modèle
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))  # (W, H) pour OpenCV

    # Normalisation + dimension batch
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Prédiction du modèle
    try:
        preds = model.predict(image)[0]  # (H, W, C) ou (H, W, 1)
    except Exception as e:
        return jsonify({"error": f"Erreur interne dans le modèle : {str(e)}"}), 500

    # Gestion multi-classes vs binaire
    if len(preds.shape) == 3 and preds.shape[-1] > 1:
        # Sortie (H, W, nb_classes) -> on prend l'argmax
        mask = np.argmax(preds, axis=-1).astype("int32")  # (H, W)
    else:
        # Cas binaire (H, W, 1) ou (H, W)
        mask = preds
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype("int32")  # 0 ou 1

    # APPLICATION DE LA COLORMAP -> image RGB (H, W, 3)
    color_mask = apply_colormap(mask)

    # Conversion en PNG pour envoi
    pil_mask = Image.fromarray(color_mask)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


# =======================================================================
# LANCEMENT LOCAL
# =======================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
