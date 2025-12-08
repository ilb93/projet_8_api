import requests

url = "http://127.0.0.1:5000/predict"

# Mets ici le chemin complet de ton image
file_path = r"C:\Users\ouafi\OneDrive\Bureau\formation\FORMATION DATA 2\projet 8\data\P8_Cityscapes_leftImg8bit_trainvaltest\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    print("✅ Status Code:", response.status_code)
    print("📨 Response:", response.json())

except FileNotFoundError:
    print(f"❌ Fichier non trouvé : {file_path}")

except Exception as e:
    print(f"⚠️ Erreur lors de la requête : {e}")
