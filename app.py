from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Bienvenue sur l'API de segmentation d'images"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    # Ici, ajoute le traitement de l'image et la prédiction...
    return jsonify({"message": "Prédiction effectuée avec succès"})

if __name__ == "__main__":
    app.run(debug=True)
