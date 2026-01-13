# Projet 8 – API de Segmentation d’images (Cityscapes) – Flask + Heroku + S3

Cette API Flask expose un modèle de segmentation sémantique entraîné sur Cityscapes (remapping en **8 classes**) et fournit :
- une prédiction de masque à partir d’une **image uploadée**,
- une prédiction à partir d’un **ID d’image** stockée sur **AWS S3**,
- des endpoints utilitaires pour l’interface Streamlit : liste des IDs, récupération image réelle et GT.

> Déploiement cible : **Heroku** (API) + **AWS S3** (modèle + dataset)  
> Interface de démonstration : **Streamlit Cloud** (consomme l’API)

---

## 1) Fonctionnalités

✅ **Prédiction**
- `POST /predict` : upload d’une image → retourne le **mask prédit** (PNG)
- `GET /predict_by_id/<id>` : prédit à partir d’une image Cityscapes sur S3 via son ID → retourne le **mask prédit** (PNG)

✅ **Données (S3)**
- `GET /ids` : liste des IDs disponibles sur S3
- `GET /image/<id>` : retourne l’image réelle (PNG)
- `GET /gt/<id>` : retourne le mask GT **colorisé Cityscapes** (PNG)

---

## 2) Classes et format

### Remapping (8 classes)
Le modèle produit des prédictions sur 8 classes remappées :
`["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]`

### Format de sortie
- Les routes de prédiction retournent **directement une image PNG** (pas du JSON).
- Le masque est colorisé avec une palette de visualisation (seed=24) et upscalé à la taille originale de l’image.

---

## 3) Structure S3 attendue

L’API suppose l’organisation suivante dans le bucket (préfixes configurables via variables d’environnement) :

### Images (Cityscapes leftImg8bit)
leftImg8bit/<city>/<id>_leftImg8bit.png

markdown
Copier le code

### Ground Truth (Cityscapes gtFine)
L’API renvoie par défaut le GT **colorisé** :
gtfine/<city>/<id>_gtFine_color.png

yaml
Copier le code

> Exemple d’ID : `aachen_000010_000019`  
> City = `aachen`

---

## 4) Endpoints

### Healthcheck
#### `GET /`
Réponse :
```json
{"message":"API de segmentation opérationnelle"}
Liste des IDs
GET /ids
Réponse :

json
Copier le code
{"ids":["aachen_000000_000019","aachen_000001_000019", "..."]}
Image réelle
GET /image/<img_id>
Retour : image/png

Exemple :

arduino
Copier le code
GET /image/aachen_000000_000019
Mask réel (GT)
GET /gt/<img_id>
Retour : image/png (Cityscapes GT color)

Exemple :

bash
Copier le code
GET /gt/aachen_000000_000019
Prédiction par ID
GET /predict_by_id/<img_id>
Retour : image/png (mask prédit)

Exemple :

bash
Copier le code
GET /predict_by_id/aachen_000000_000019
Prédiction par upload
POST /predict
Body : multipart/form-data avec champ file

Retour : image/png (mask prédit)

Exemple curl :

bash
Copier le code
curl -X POST https://<ton-app>.herokuapp.com/predict \
  -F "file=@/path/to/image.png" \
  --output mask.png
5) Variables d’environnement (Heroku Config Vars)
Obligatoires
AWS_REGION (ex: eu-central-1)

S3_BUCKET : bucket contenant le modèle

S3_MODEL_KEY : clé S3 du modèle .h5 (ex: vgg_unet_trained_model.h5)

Données (optionnelles)
S3_DATA_BUCKET : bucket dataset (si différent de S3_BUCKET)

S3_LEFT_PREFIX : préfixe images (défaut leftImg8bit/)

S3_GT_PREFIX : préfixe GT (défaut gtfine/)

6) Installation locale
Prérequis
Python 3.10+ recommandé

Un accès AWS IAM permettant s3:GetObject sur le bucket

Installation
bash
Copier le code
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
Lancer l’API
bash
Copier le code
export AWS_REGION=eu-central-1
export S3_BUCKET=...
export S3_MODEL_KEY=...
export S3_DATA_BUCKET=...
python app.py
L’API démarre par défaut sur :

cpp
Copier le code
http://127.0.0.1:5000
7) Déploiement sur Heroku (résumé)
Définir les Config Vars (Settings → Config Vars)

Push du code vers le repo connecté à Heroku (ou déploiement GitHub auto)

Le modèle est téléchargé au boot dans /tmp/model.h5

8) Interface Streamlit (consommation de l’API)
L’application Streamlit consomme les routes suivantes :

/ids → liste IDs

/image/<id> → image réelle

/gt/<id> → mask GT colorisé

/predict_by_id/<id> → mask prédit

Objectif : démontrer le workflow complet “sélection ID → appel API → affichage résultats”.

9) Notes importantes
Le GT affiché via /gt/<id> est Cityscapes color (gtFine_color.png) : lisible et conforme pour la visualisation.

Les couleurs du mask prédit sont une palette de visualisation (seed=24) et peuvent différer de la palette officielle Cityscapes.
Cela ne change pas le contenu sémantique, uniquement l’affichage.
