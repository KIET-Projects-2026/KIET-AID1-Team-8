from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# ------------------ CONFIG ------------------

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ DEVICE ------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODEL (SAFE) ------------------

print("⏳ Loading image caption model...")

try:
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )

    model.to(device)
    model.eval()

    # VERY IMPORTANT
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

# ------------------ IMAGE CAPTION API ------------------

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if model is None:
        return jsonify({"error": "Model not loaded. Check internet connection."})

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Empty filename"})

    filename = secure_filename(image_file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(path)

    # Load & preprocess image
    image = Image.open(path).convert("RGB")
    pixel_values = feature_extractor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=20,
            num_beams=4,
            early_stopping=True
        )

    caption = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )

    if caption.strip() == "":
        caption = "Unable to generate caption for this image"

    # Generate audio
    audio_path = os.path.join(STATIC_FOLDER, "caption.mp3")
    tts = gTTS(text=caption, lang="en")
    tts.save(audio_path)

    return jsonify({
        "caption": caption,
        "audio": "/static/caption.mp3"
    })

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(debug=True)
