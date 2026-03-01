from flask import Flask, request
import requests
import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# ====== LOAD FROM ENVIRONMENT (GitHub Secrets or Local) ======
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
# ===============================================================

DISEASE_MANAGEMENT = {
    "Bacterial Leaf Blight": "⚠️ *Bacterial Leaf Blight (பாக்டீரியா இலை கருகல் நோய்)* symptoms are visible. / அறிகுறிகள் தெரிகின்றன.\n\n✅ *Chemical Management :*\nMix Streptomycin sulphate + Tetracycline combination (e.g., Streptocycline) @ 120g/acre + Copper oxychloride @ 500g/acre in water and spray.\n\n✅ *இரசாயன மேலாண்மை :*\nஸ்ட்ரெப்டோமைசின் சல்பேட் + டெட்ராசைக்ளின் கலவை (எ.கா: ஸ்ட்ரெப்டோசைக்ளின்) ஏக்கருக்கு 120 கிராம் மற்றும் காப்பர் ஆக்சி குளோரைடு ஏக்கருக்கு 500 கிராம் என்ற அளவில் தண்ணீரில் கலந்து தெளிக்கவும்.\n\n🌱 *Organic Management :*\nSpray 20% fresh cow dung extract or spray 0.5% Neem oil.\n\n🌱 *இயற்கை மேலாண்மை (Tamil):*\n20% புதிய சாணக் கரைசல் அல்லது 0.5% வேப்ப எண்ணெய் தெளிக்கவும்.",

    "Brown Spot": "⚠️ *Brown Spot (பழுப்பு புள்ளி நோய்)* is affecting the plant. / பயிரை பாதித்துள்ளது.\n\n✅ *Chemical Management :*\nMix Mancozeb 2.0 g/litre or Edifenphos 1 ml/litre of water and spray.\n\n✅ *இரசாயன மேலாண்மை :*\nமேன்கோசெப் 2.0 கிராம்/லிட்டர் அல்லது எடிஃபென்ஃபோஸ் 1 மி.லி/லிட்டர் தண்ணீரில் கலந்து தெளிக்கவும்.\n\n🌱 *Organic Management :*\nSpray Pseudomonas fluorescens @ 1 kg/acre or 5% Neem Seed Kernel Extract (NSKE).\n\n🌱 *இயற்கை மேலாண்மை :*\nசூடோமோனாஸ் புளோரசன்ஸ் ஏக்கருக்கு 1 கிலோ அல்லது 5% வேப்பங்கொட்டை சாறு தெளிக்கவும்.",

    "Healthy Rice Leaf": "🌿 Super! Your paddy crop is healthy. There is no disease. Best wishes for a good yield!\n\n🌿 சூப்பர்! உங்கள் நெற்பயிர் ஆரோக்கியமாக உள்ளது. எந்த நோயும் இல்லை. நல்ல மகசூல் கிடைக்க வாழ்த்துக்கள்!",

    "Leaf Blast": "⚠️ *Leaf Blast (குலை நோய்)* symptoms are present. / அறிகுறிகள் உள்ளன.\n\n✅ *Chemical Management :*\nSpray Tricyclazole 75 WP @ 200g/acre. Avoid water stagnation in the field.\n\n✅ *இரசாயன மேலாண்மை :*\nட்ரைசைக்ளோசோல் 75 WP ஏக்கருக்கு 200 கிராம் தெளிக்கவும். வயலில் தண்ணீர் தேங்குவதைத் தவிர்க்கவும்.\n\n🌱 *Organic Management :*\nSpray Pseudomonas fluorescens @ 1 kg/acre. Avoid excessive application of nitrogen fertilizers.\n\n🌱 *இயற்கை மேலாண்மை :*\nசூடோமோனாஸ் புளோரசன்ஸ் ஏக்கருக்கு 1 கிலோ என்ற அளவில் தெளிக்கவும். தழைச்சத்து உரங்களை அதிகமாகப் பயன்படுத்துவதைத் தவிர்க்கவும்.",

    "Leaf scald": "⚠️ *Leaf Scald (இலை கருகல் / வெளிறிய நோய்)* symptoms are visible. / அறிகுறிகள் தெரிகின்றன.\n\n✅ *Chemical Management :*\nMix Hexaconazole 5 EC @ 2 ml/litre or Propiconazole 25 EC @ 1 ml/litre in water and spray.\n\n✅ *இரசாயன மேலாண்மை :*\nஹெக்ஸாகோனசோல் 5 EC 2 மி.லி/லிட்டர் அல்லது ப்ரோபிகோனசோல் 25 EC 1 மி.லி/லிட்டர் தண்ணீரில் கலந்து தெளிக்கவும்.\n\n🌱 *Organic Management :*\nSpray 5% Neem Seed Kernel Extract (NSKE) or 3% Neem oil. Maintain proper plant spacing.\n\n🌱 *இயற்கை மேலாண்மை :*\n5% வேப்பங்கொட்டை சாறு அல்லது 3% வேப்ப எண்ணெய் தெளிக்கவும். பயிர்களுக்கு இடையே சரியான இடைவெளியை பராமரிக்கவும்.",

    "Sheath Blight": "⚠️ *Sheath Blight (இலை உறை அழுகல் நோய்)* is affecting the plant. / பயிரை பாதித்துள்ளது.\n\n✅ *Chemical Management :*\nSpray Carbendazim 50 WP @ 200g/acre or Propiconazole 25 EC @ 200 ml/acre. Maintain proper plant spacing.\n\n✅ *இரசாயன மேலாண்மை :*\nகார்பென்டாசிம் 50 WP ஏக்கருக்கு 200 கிராம் அல்லது ப்ரோபிகோனசோல் 25 EC ஏக்கருக்கு 200 மி.லி தெளிக்கவும். பயிர்களுக்கு இடையே சரியான இடைவெளியை பராமரிக்கவும்.\n\n🌱 *Organic Management :*\nApply Pseudomonas fluorescens or Trichoderma viride @ 1 kg/acre mixed with 20 kg of farmyard manure to the soil.\n\n🌱 *இயற்கை மேலாண்மை :*\nசூடோமோனாஸ் புளோரசன்ஸ் அல்லது ட்ரைக்கோடெர்மா விரிடி ஏக்கருக்கு 1 கிலோ வீதம் 20 கிலோ தொழுஉரத்துடன் கலந்து வயலில் இடவும்."
}


def predict_paddy_disease(image_path, model_path='keras_model.h5', labels_path='labels.txt'):
    np.set_printoptions(suppress=True)

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(labels_path, "r") as f:
        class_names = f.readlines()

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

    if mode and token:
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

            send_whatsapp_message(sender, "🔍 Image received! Analyzing paddy leaf... Please wait.")

            try:
                predicted_class, confidence = predict_paddy_disease(local_image_path)

                advice = DISEASE_MANAGEMENT.get(
                    predicted_class,
                    "Diagnosis complete, but specific management not found. Please consult an expert."
                )

                reply_text = (
                    f"📊 *Diagnosis:* {predicted_class}\n"
                    f"🎯 *Accuracy:* {confidence * 100:.1f}%\n\n"
                    f"{advice}"
                )

            except Exception as e:
                print("Prediction Error:", e)
                reply_text = "Sorry, model process pandrathula oru error vandhuduchu. Please try again with a clear image."

        else:
            reply_text = "🌱 Please send a clear image of the paddy leaf to detect disease."

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

