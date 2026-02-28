from flask import Flask, request
import requests
import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# ====== LOAD FROM ENVIRONMENT VARIABLES ======
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
# =============================================

# ====== LOAD MODEL ONLY ONCE (IMPORTANT) ======
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = f.readlines()
# ==============================================

DISEASE_MANAGEMENT = {
    "Bacterial Leaf Blight": "📊 Diagnosis: Bacterial Leaf Blight\n🎯 Accuracy: {accuracy:.1f}%\n\n⚠️ Bacterial Leaf Blight (பாக்டீரியா இலை கருகல்) affect aagirukku.\n\n✅ Management:\nUrea use panratha kurainga. Field-la irukka water-a 3-4 days vadichidunga. 10 litre water-la Streptocycline 1g + Copper oxychloride 30g mix panni spray pannunga.",
    
    "Brown Spot": "📊 Diagnosis: Brown Spot\n🎯 Accuracy: {accuracy:.1f}%\n\n⚠️ Brown Spot (பழுப்பு புள்ளி நோய்) affect aagirukku.\n\n✅ Management:\nSoil-la potash sattu athigama podunga. Field-a weed illama vachikonga. Mancozeb 2.5 g/litre or Propiconazole 1 ml/litre of water-la mix panni spray pannunga.",
    
    "Healthy Rice Leaf": "📊 Diagnosis: Healthy Rice Leaf\n🎯 Accuracy: {accuracy:.1f}%\n\n✅ Unga crop healthy-a irukku! Regular-a field-a monitor pannunga. Correct-ana alavula fertilizers use pannunga. 🌿",
    
    "Leaf Blast": "📊 Diagnosis: Leaf Blast\n🎯 Accuracy: {accuracy:.1f}%\n\n⚠️ Leaf Blast (இலை குலை நோய்) affect aagirukku.\n\n✅ Management:\nExcess urea poduratha thavirthudunga. Varappu (bunds) clean-a vachikonga. Tricyclazole 75 WP 0.6 g/litre or Pseudomonas fluorescens 5 g/litre of water-la mix panni spray pannunga.",
    
    "Leaf scald": "📊 Diagnosis: Leaf Scald\n🎯 Accuracy: {accuracy:.1f}%\n\n⚠️ Leaf Scald (இலை கருகல் நோய்) affect aagirukku.\n\n✅ Management:\nAdarthiyaa nadavu seirathaiyum, excess urea podurathaiyum thavirthudunga. Hexaconazole or Propiconazole 1 ml/litre of water-la mix panni spray pannunga.",
    
    "Sheath Blight": "📊 Diagnosis: Sheath Blight\n🎯 Accuracy: {accuracy:.1f}%\n\n⚠️ Sheath Blight (இலை உறை கருகல் நோய்) affect aagirukku.\n\n✅ Management:\nPlants-ku naduvula correct-ana gap maintain pannunga. Thodarndhu thanneer thenguratha thavirthudunga. Validamycin 2 ml/litre or Hexaconazole 2 ml/litre of water-la mix panni thoorla (base) padura mathiri spray pannunga."
}

def predict_paddy_disease(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()

    # Remove any numeric prefix from the labels (e.g., "0 Brown Spot" -> "Brown Spot")
    if " " in class_name:
        class_name = class_name.split(" ", 1)[1]

    confidence_score = prediction[0][index]
    return class_name, confidence_score


@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    try:
        # Navigate through the WhatsApp webhook payload
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]

        if message["type"] == "image":
            media_id = message["image"]["id"]
            
            # Send an immediate processing message so the user knows it's working
            send_whatsapp_message(sender, "🔍 Image received! Analyzing...")

            # Get image URL and download it
            media_url = get_media_url(media_id)
            download_image(media_url, media_id)
            local_image_path = f"images/{media_id}.jpg"

            # Predict disease
            predicted_class, confidence = predict_paddy_disease(local_image_path)
            accuracy_percent = confidence * 100

            # Format the reply using our Tanglish dictionary
            if predicted_class in DISEASE_MANAGEMENT:
                reply_text = DISEASE_MANAGEMENT[predicted_class].format(accuracy=accuracy_percent)
            else:
                # Fallback in case the model predicts something outside the dictionary
                reply_text = (
                    f"📊 Diagnosis: {predicted_class}\n"
                    f"🎯 Accuracy: {accuracy_percent:.1f}%\n\n"
                    f"⚠️ Please consult an agriculture expert for management."
                )

        else:
            reply_text = "🌱 Please send a clear paddy leaf image."

        # Send the final result back to the user
        send_whatsapp_message(sender, reply_text)

    except Exception as e:
        print("Webhook Error:", e)

    return "OK", 200


def get_media_url(media_id):
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)
    return response.json()["url"]


def download_image(url, media_id):
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)

    if not os.path.exists("images"):
        os.makedirs("images")

    with open(f"images/{media_id}.jpg", "wb") as f:
        f.write(response.content)


def send_whatsapp_message(to, text):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    requests.post(url, headers=headers, json=payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
