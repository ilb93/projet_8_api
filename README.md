✅ 1. FICHIERS À SUPPRIMER (inutile / bruit / pas livrable)
Supprime complètement :
Fichier / Dossier	Pourquoi le supprimer
__pycache__/	Fichiers temporaires Python → jamais versionnés
temp_results/	Résultats temporaires → inutile en prod
segmentation_result.png	Exemple de sortie → pas nécessaire dans le repo
mask_raw.png	Idem, inutile
mask_colorized.png	Test local → pollue le repo
.python-version	Spécifique à pyenv, pas demandé dans le projet
test_api.py	Fichier de debug interne
Toute image générée automatiquement	À retirer sauf si utile à la doc

⚠️ Attention : ne supprime PAS :

app.py

streamlit_app.py

requirements.txt

Procfile

unet_final_model.h5

C'est le cœur du livrable.

✅ 2. FICHIERS À GARDER (obligatoires pour la soutenance)
À conserver absolument :
Fichier	Rôle
app.py	API Flask / FastAPI du Projet 8
streamlit_app.py	Interface utilisateur
requirements.txt	Dépendances pour déploiement
Procfile	Déploiement Heroku
unet_final_model.h5	Modèle pré-entraîné obligatoire pour l’évaluation
✅ 3. README COMPLET & PROFESSIONNEL À COLLER DIRECTEMENT DANS GITHUB

Voici un README formaté prêt pour GitHub (Markdown).
Tu as juste à copier-coller dans ton README.md :

🧠 Projet 8 – Segmentation d’Images avec VGG-UNet
API + Interface Streamlit

Ce dépôt contient l’API de prédiction et l’application Streamlit permettant de tester un modèle de segmentation sémantique entraîné sur le dataset Cityscapes (projet OpenClassrooms – Data Scientist).

🎯 Objectif du projet

Développer un modèle capable de segmenter des scènes urbaines en 8 catégories principales :

void

flat

construction

object

nature

sky

human

vehicle

Les 34 classes originales de Cityscapes ont été remappées selon les consignes du projet.

La solution finale inclut :

un modèle VGG-UNet entraîné sur 2 303 images,

une API fournissant la segmentation au format PNG,

une interface Streamlit permettant d’envoyer une image et d'afficher le masque segmenté.

🚀 Architecture du dépôt
.
├── app.py                → API Flask pour la segmentation
├── streamlit_app.py      → Interface utilisateur
├── requirements.txt      → Dépendances Python
├── Procfile              → Configuration Heroku
├── unet_final_model.h5   → Modèle entraîné
└── README.md             → Documentation

🧩 Fonctionnement
1. L’API (app.py)

Elle permet :

de recevoir une image (upload),

de la redimensionner / normaliser,

d’exécuter le modèle VGG-UNet,

de retourner un masque segmenté (8 classes).

Endpoint principal :

POST /predict


Entrée : image (.jpg / .png)
Sortie : masque segmenté colorisé

2. Interface Streamlit (streamlit_app.py)

Permet à l’utilisateur de :

charger une image locale,

visualiser l’image originale,

afficher le masque segmenté,

lire la légende des 8 classes.

🧠 Modèle utilisé

Architecture : VGG-UNet (Encoder VGG16 + Decoder U-Net)

Taille d’entrée : 256×512

Fonction de perte : categorical_crossentropy

Métriques : IoU, Dice, Accuracy

Optimiseur : Adam

Entraînement réalisé sur :

2 303 images train

403 images validation

3 epochs (contraintes CPU)

🛠️ Installation
1. Cloner le projet
git clone https://github.com/ilb93/projet_8_api.git
cd projet_8_api

2. Installer les dépendances
pip install -r requirements.txt

3. Lancer l’API
python app.py

4. Lancer l'application Streamlit
streamlit run streamlit_app.py

🌐 Déploiement Heroku

Le projet inclut :

Procfile

requirements.txt

Heroku détecte automatiquement Flask et exécute :

web: gunicorn app:app

📚 Ressources complémentaires

Notebook d’entraînement du modèle

Note technique (10 pages)

Pipeline d’augmentation configuré mais non activé dans l’entraînement final

Explications détaillées sur le remapping Cityscapes
