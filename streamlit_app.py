import io

import requests
import streamlit as st
from PIL import Image

# ======================================================================
# CONFIG
# ======================================================================

API_URL = "https://projet8-api-mourad-c5ec745a525e.herokuapp.com/predict"

st.set_page_config(
    page_title="Projet 8 - Segmentation d'images",
    layout="centered",
)

# ======================================================================
# UI
# ======================================================================

st.title("Projet 8 – Segmentation d'images (API Heroku)")
st.write("Choisissez une image et cliquez sur **Segmenter** pour envoyer la requête à l’API déployée sur Heroku.")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Uploader une image",
    type=["jpg", "jpeg", "png"],
    help="Utilise une image du dataset Cityscapes par exemple."
)

if uploaded_file is not None:
    # Affichage de l'image originale
    st.image(
        uploaded_file,
        caption="Image originale",
        use_container_width=True,   # remplace use_column_width (déprécié)
    )

    if st.button("Segmenter"):
        # Préparation des données pour l'API
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/png",
            )
        }

        with st.spinner("Envoi de l'image à l'API Heroku..."):
            try:
                response = requests.post(API_URL, files=files, timeout=60)
            except Exception as e:
                st.error("Erreur lors de l'appel à l'API.")
                st.code(str(e))
            else:
                if response.status_code == 200:
                    st.success("Segmentation réussie !")

                    # L'API retourne directement une image PNG (mask segmenté)
                    mask_bytes = response.content
                    mask_image = Image.open(io.BytesIO(mask_bytes))

                    st.image(
                        mask_image,
                        caption="Masque segmenté",
                        use_container_width=True,
                    )
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    # Affichage du contenu brut renvoyé par l'API (texte / HTML / JSON)
                    try:
                        st.code(response.json(), language="json")
                    except Exception:
                        st.code(response.text)
else:
    st.info("Commence par uploader une image pour activer le bouton **Segmenter**.")
