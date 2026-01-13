# Projet 8 – Application Streamlit (Interface de test API de segmentation)

Cette application **Streamlit** est l’interface de démonstration du Projet 8.  
Elle consomme l’API Flask déployée sur le cloud (Heroku) afin de :

✅ afficher la **liste des IDs d’images disponibles** (dataset Cityscapes sur S3)  
✅ permettre la **sélection d’un ID**  
✅ afficher :
- l’**image réelle**
- le **mask réel (Ground Truth)**
- le **mask prédit** (via appel à l’API)

> Objectif : fournir une interface simple pour tester l’API et illustrer les résultats auprès de collègues / évaluateurs.

---

## 1) Fonctionnalités

### Mode 1 — Sélection par ID (conforme consigne)
- Récupère la liste des IDs via `GET /ids`
- Affiche l’image réelle via `GET /image/<id>`
- Affiche le mask réel via `GET /gt/<id>`
- Lance la prédiction via `GET /predict_by_id/<id>`
- Affiche le mask prédit

### Mode 2 — Upload (secours)
- Permet d’uploader une image locale
- Appelle l’API via `POST /predict`
- Affiche le mask prédit retourné par l’API

---

## 2) Pré-requis

- Python 3.10+ recommandé
- Une API fonctionnelle accessible publiquement (ex : Heroku)
- L’API doit exposer au minimum :
  - `GET /ids`
  - `GET /image/<id>`
  - `GET /gt/<id>`
  - `GET /predict_by_id/<id>`
  - `POST /predict`

---

## 3) Installation locale

### Cloner le projet
```bash
git clone https://github.com/ilb93/projet_8_api.git
cd projet_8_api
Installer les dépendances Streamlit
Crée un fichier requirements_streamlit.txt (ou requirements.txt dédié Streamlit) contenant :

txt
Copier le code
streamlit
requests
pillow
Puis installe :

bash
Copier le code
pip install -r requirements_streamlit.txt
4) Lancer l’application en local
Dans le dossier où se trouve streamlit_app.py :

bash
Copier le code
streamlit run streamlit_app.py
L’application sera disponible sur :

arduino
Copier le code
http://localhost:8501
5) Configuration de l’URL API
Dans le fichier Streamlit, l’URL de l’API est définie via :

python
Copier le code
API_URL = st.secrets.get("API_URL", "https://TON-API.herokuapp.com")
Option A — En local (simple)
Tu peux remplacer directement dans le code :

python
Copier le code
API_URL = "https://projet8-api-mourad-xxxx.herokuapp.com"
Option B — Sur Streamlit Cloud (recommandé)
Dans Streamlit Cloud → App settings → Secrets, ajouter :

toml
Copier le code
API_URL = "https://projet8-api-mourad-xxxx.herokuapp.com"
6) Utilisation
Mode ID (dataset sur S3)
Sélectionner un ID dans la liste déroulante

L’application affiche automatiquement :

Image réelle

Mask réel (GT)

Cliquer sur Segmenter (par ID)

Le mask prédit s’affiche à droite

Mode Upload (secours)
Uploader une image .png/.jpg

Cliquer sur Segmenter

Le mask prédit s’affiche

7) Déploiement Streamlit Cloud
Pousser le code sur GitHub

Aller sur https://streamlit.io/cloud

Connecter ton repo GitHub

Sélectionner le fichier principal :

streamlit_app.py

Ajouter dans les Secrets :

toml
Copier le code
API_URL = "https://projet8-api-mourad-xxxx.herokuapp.com"
Déployer

8) Notes importantes
Le mask réel (GT) affiché est le fichier Cityscapes colorisé (gtFine_color.png) → meilleur rendu visuel.

Le mask prédit est colorisé avec une palette de visualisation du projet.
Les couleurs peuvent être différentes du GT, mais le but est d’illustrer la segmentation.

9) Exemple d’architecture (résumé)
Streamlit (Cloud)
⬇️ appels HTTP
API Flask (Heroku)
⬇️ lecture des fichiers
AWS S3 (images + GT + modèle)

