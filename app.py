import os
from io import BytesIO

# Réduire le bruit des logs TensorFlow (à faire avant l'import tf)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, jsonify, request, send_file
from PIL import Image


# =======================================================================
# CONFIG GLOBALE
# =======================================================================

# Taille d'entrée attendue par le modèle : 256 (H) x 512 (W)
IMG_HEIGHT = 256
IMG_WIDTH = 512

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

# Nom du fichier **local** du modèle (dans le conteneur Heroku)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "vgg_unet_trained_model.h5"
)

# URL de téléchargement direct depuis ton Google Drive
MODEL_URL = (
    "https://drive.google.com/uc"
    "?export=download&id=1k3wtDmMviqrysyw1dwzpJIUQ5J0Of_yR"
)


# =======================================================================
# TELECHARGEMENT + CHARGEMENT DU MODELE
# =======================================================================

def download_model_if_needed():
    """
    Télécharge le modèle depuis Google Drive s'il n'existe pas encore
    sur le disque (dans le conteneur Heroku).
    """
    if os.path.exists(MODEL_PATH):
        print(f"✅ Modèle déjà présent : {MODEL_PATH}")
        return

    print("⬇️ Téléchargement du modèle depuis Google Drive ...")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()

    # Écriture par chunks pour éviter de tout garder en RAM
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✅ Modèle téléchargé et sauvegardé : {MODEL_PATH}")


# On s'assure que le fichier modèle est présent, puis on le charge
download_model_if_needed()

print(f"✅ Chargement du modèle depuis : {MODEL_PATH}")
# compile=False pour éviter les problèmes de custom loss (combined_loss, etc.)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modèle chargé en mémoire avec succès !")


# =======================================================================
# FONCTIONS UTILITAIRES
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

    # Tous les labels non définis restent noirs (0, 0, 0)
    return color_mask


# =======================================================================
# ROUTE DE SANTE / TEST
# =======================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de segmentation projet 8 opérationnelle"})


# =======================================================================
# ROUTE DE PREDICTION
# =======================================================================

@app.route("/predict", methods=["POST"])
def predict():
    # Vérification de la présence du fichier
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée (clé 'file' manquante)"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Décodage de l'image (OpenCV -> BGR)
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Format d'image non supporté"}), 400

    # Redimensionnement au format attendu par le modèle
    # OpenCV attend (width, height)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalisation et ajout de la dimension batch
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    # Prédiction
    try:
        preds = model.predict(image)[0]  # (H, W, C) ou (H, W, 1)
    except Exception as e:
        return jsonify({"error": f"Erreur interne dans le modèle : {str(e)}"}), 500

    # Gestion multi-classes vs binaire
    if preds.ndim == 3 and preds.shape[-1] > 1:
        # Sortie (H, W, nb_classes) -> argmax
        mask = np.argmax(preds, axis=-1).astype("int32")  # (H, W)
    else:
        # Cas binaire (H, W, 1) ou (H, W)
        mask = preds
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0.5).astype("int32")  # 0 ou 1

    # Application de la colormap
    color_mask = apply_colormap(mask)

    # Conversion en PNG pour renvoi HTTP
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
    # debug=True pratique en local, Heroku s'en fiche
    app.run(host="0.0.0.0", port=port, debug=True)
