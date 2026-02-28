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
    "Bacterial Leaf Blight": "Reduce urea (nitrogen) fertilizer. Drain field water for 3-4 days to stop the spread. Spray Streptocycline (1g) + Copper oxychloride (30g) in 10L of water.\n\nயூரியா இடுவதைக் குறைக்கவும். நோய் பரவலைத் தடுக்க வயலில் உள்ள நீரை 3-4 நாட்களுக்கு வடிக்கவும். 10 லிட்டர் தண்ணீருக்கு 1 கிராம் ஸ்ட்ரெப்டோசைக்ளின் + 30 கிராம் காப்பர் ஆக்ஸிகுளோரைடு கலந்து தெளிக்கவும்.",
    
    "Brown Spot": "Ensure proper soil nutrition, especially potash. Keep the field weed-free. Spray Mancozeb (2.5g/Liter) or Propiconazole (1ml/Liter).\n\nமண்ணின் ஊட்டச்சத்தை மேம்படுத்தவும், குறிப்பாக சாம்பல் சத்து (Potash) இடுங்கள். வயலை களைகளின்றி வைக்கவும். 1 லிட்டர் தண்ணீருக்கு 2.5 கிராம் மான்கோசெப் அல்லது 1 மி.லி புரொப்பிகோனசோல் கலந்து தெளிக்கவும்.",
    
    "Healthy Rice Leaf": "Your crop is healthy! Continue regular field monitoring and maintain balanced fertilizers. Keep up the good work! 🌿\n\nஉங்கள் நெற்பயிர் ஆரோக்கியமாக உள்ளது! தொடர்ந்து வயலை கண்காணிக்கவும் மற்றும் சரியான அளவில் உரங்களை இடவும். வாழ்த்துகள்! 🌿",
    
    "Leaf Blast": "Avoid excess urea and delay top-dressing. Keep the field bunds clean. Spray Tricyclazole 75 WP (0.6g/Liter) or Pseudomonas fluorescens (5g/Liter).\n\nஅதிகப்படியான யூரியா இடுவதைத் தவிர்க்கவும். வரப்புகளை சுத்தமாக வைக்கவும். 1 லிட்டர் தண்ணீருக்கு 0.6 கிராம் ட்ரைசைக்லாசோல் அல்லது 5 கிராம் சூடோமோனாஸ் கலந்து தெளிக்கவும்.",
    
    "Leaf scald": "Use disease-free seeds for the next season. Avoid thick planting and high nitrogen. Spray Hexaconazole or Propiconazole (1ml/Liter) if symptoms are severe.\n\nஅடர்த்தியாக நடவு செய்வதையும், அதிக யூரியா இடுவதையும் தவிர்க்கவும். நோய் தீவிரமாக இருந்தால் 1 லிட்டர் தண்ணீருக்கு 1 மி.லி ஹெக்ஸாகோனசோல் அல்லது புரொப்பிகோனசோல் கலந்து தெளிக்கவும்.",
    
    "Sheath Blight": "Maintain proper spacing between plants. Avoid continuous waterlogging. Spray Validamycin (2ml/Liter) or Hexaconazole (2ml/Liter) aiming at the base of the plant.\n\nபயிர்களுக்கு இடையே சரியான இடைவெளியை பராமரிக்கவும். வயலில் தொடர்ந்து தண்ணீர் தேங்குவதைத் தவிர்க்கவும். 1 லிட்டர் தண்ணீருக்கு 2 மி.லி வேலிடமைசின் அல்லது ஹெக்ஸாகோனசோல் கலந்து தூரில் படும்படி தெளிக்கவும்."
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
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]

        if message["type"] == "image":
            media_id = message["image"]["id"]
            media_url = get_media_url(media_id)

            download_image(media_url, media_id)
            local_image_path = f"images/{media_id}.jpg"

            send_whatsapp_message(sender, "🔍 Image received! Analyzing...")

            predicted_class, confidence = predict_paddy_disease(local_image_path)
            advice = DISEASE_MANAGEMENT.get(predicted_class, "Consult expert.")

            reply_text = (
                f"📊 Diagnosis: {predicted_class}\n"
                f"🎯 Accuracy: {confidence * 100:.1f}%\n\n"
                f"{advice}"
            )

        else:
            reply_text = "🌱 Please send a clear paddy leaf image."

        send_whatsapp_message(sender, reply_text)

    except Exception as e:
        print("Error:", e)

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

