import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "L'API fonctionne correctement !"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Utilisation du port de Heroku
    app.run(host="0.0.0.0", port=port, debug=True)
