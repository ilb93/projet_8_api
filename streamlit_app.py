import io
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# =========================================================
# CONFIG
# =========================================================
# Mets ici ton domaine (sans /predict)
API_BASE = st.secrets.get("API_BASE", "https://projet8-api-mourad-c5ec745a525e.herokuapp.com")

# Endpoints
IDS_URL = f"{API_BASE}/ids"
IMAGE_URL = f"{API_BASE}/image"          # /<id>
GT_URL = f"{API_BASE}/gt"                # /<id>
PRED_BY_ID_URL = f"{API_BASE}/predict_by_id"  # /<id>
PRED_UPLOAD_URL = f"{API_BASE}/predict"  # POST upload

st.set_page_config(page_title="Projet 8 - Segmentation d'images", layout="wide")

# =========================================================
# Helpers
# =========================================================
def fetch_png(url: str, timeout: int = 60) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

# =========================================================
# UI
# =========================================================
st.title("Projet 8 – Segmentation d'images (API Heroku)")
st.write("Interface de test : **liste d’IDs**, affichage **image réelle / GT / prédiction** via API.")

tab1, tab2 = st.tabs(["Mode ID (conforme consigne)", "Mode Upload (secours)"])

# =========================================================
# TAB 1 : MODE ID
# =========================================================
with tab1:
    st.subheader("1) Sélection par ID (dataset sur S3)")

    # Charger IDs
    ids_list = []
    try:
        ids_resp = requests.get(IDS_URL, timeout=30)
        ids_resp.raise_for_status()
        data = ids_resp.json()
        ids_list = data.get("ids", [])
    except Exception as e:
        st.error("Impossible de récupérer la liste des IDs via /ids.")
        st.code(str(e))

    if not ids_list:
        st.warning("Aucun ID trouvé. Vérifie que l’API expose bien /ids et que S3 est accessible.")
    else:
        img_id = st.selectbox("Choisir un ID d'image disponible", ids_list)

        col1, col2, col3 = st.columns(3)

        # Image réelle
        with col1:
            st.markdown("### Image réelle")
            try:
                img_pil = fetch_png(f"{IMAGE_URL}/{img_id}", timeout=60)
                st.image(img_pil, use_container_width=True)
            except Exception as e:
                st.error("Erreur chargement image réelle (/image/<id>).")
                st.code(str(e))

        # GT
        with col2:
            st.markdown("### Mask réel (GT)")
            try:
                gt_pil = fetch_png(f"{GT_URL}/{img_id}", timeout=60)
                st.image(gt_pil, use_container_width=True)
            except Exception as e:
                st.error("Erreur chargement GT (/gt/<id>).")
                st.code(str(e))

        # Prediction
        with col3:
            st.markdown("### Mask prédit")
            if st.button("Segmenter (par ID)", type="primary"):
                try:
                    pred_pil = fetch_png(f"{PRED_BY_ID_URL}/{img_id}", timeout=120)
                    st.image(pred_pil, use_container_width=True)
                    st.success("Prédiction réussie.")
                except Exception as e:
                    st.error("Erreur prédiction (/predict_by_id/<id>).")
                    st.code(str(e))

# =========================================================
# TAB 2 : MODE UPLOAD (secours)
# =========================================================
with tab2:
    st.subheader("2) Upload (secours)")
    st.write("Ce mode envoie une image uploadée vers `POST /predict` et affiche le mask renvoyé.")

    uploaded_file = st.file_uploader(
        "Uploader une image",
        type=["jpg", "jpeg", "png"],
        help="Ex: une image Cityscapes."
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image originale", use_container_width=True)

        if st.button("Segmenter (upload)"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "image/png",
                )
            }

            with st.spinner("Appel API /predict..."):
                try:
                    response = requests.post(PRED_UPLOAD_URL, files=files, timeout=120)
                except Exception as e:
                    st.error("Erreur lors de l'appel à l'API.")
                    st.code(str(e))
                else:
                    if response.status_code == 200:
                        try:
                            mask_image = Image.open(io.BytesIO(response.content))
                            st.image(mask_image, caption="Masque segmenté (API)", use_container_width=True)
                            st.success("Segmentation réussie !")
                        except Exception:
                            st.error("Réponse API reçue mais impossible d'ouvrir l'image PNG.")
                            st.code(response.text)
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                        j = safe_json(response)
                        if j is not None:
                            st.json(j)
                        else:
                            st.code(response.text)
    else:
        st.info("Upload une image pour activer le bouton.")
