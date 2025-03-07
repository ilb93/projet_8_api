import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "L'API fonctionne correctement !"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Aucune donnée reçue"}), 400
    return jsonify({"message": "Prédiction reçue", "data": data})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Heroku impose cette variable
    app.run(host="0.0.0.0", port=port, debug=False)
