import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "livestock_app.db"

MODEL_PATH = MODEL_DIR / "livestock_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

LOW_CONFIDENCE_LIMIT = 0.50
HIGH_CONFIDENCE_LIMIT = 0.75

app = Flask(__name__)
app.secret_key = "local-project-secret-key"
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# General precautions only—no medicine or diagnosis advice.
GUIDANCE = {
    "foot_mouth_disease": {
        "title": "Possible foot-and-mouth disease (FMD)",
        "summary": "A photo cannot confirm FMD. Contact a veterinarian or local animal-health authority promptly.",
        "precautions": [
            "Avoid moving the suspected animal or shared equipment until you receive veterinary advice.",
            "Keep people and equipment from moving between affected and healthy animals where practical.",
            "Do not try to treat or diagnose FMD using this model result.",
        ],
        "source_url": "https://www.woah.org/en/disease/foot-and-mouth-disease/",
        "source_name": "WOAH",
    },
    "healthy": {
        "title": "No listed condition flagged",
        "summary": "The model selected the healthy class. This does not rule out illness.",
        "precautions": [
            "Continue routine care and veterinary checkups.",
            "Contact a veterinarian if you notice symptoms or behavior changes.",
            "Use a clear, well-lit photo if screening another image.",
        ],
        "source_url": None,
        "source_name": None,
    },
    "ibk_disease": {
        "title": "Possible infectious bovine keratoconjunctivitis (IBK)",
        "summary": "IBK affects the eye. A veterinarian should assess eye pain, cloudiness, or discharge.",
        "precautions": [
            "Contact a veterinarian promptly if the eye looks painful, cloudy, or has discharge.",
            "Provide shade and reduce exposure to dust and flies where practical.",
            "Do not use unprescribed eye drops or medicines.",
        ],
        "source_url": "https://www.msdvetmanual.com/eye-diseases-and-disorders/infectious-ophthalmia-infectiousophthalmia/infectious-keratoconjunctivitis-in-cattle-and-small-ruminants",
        "source_name": "MSD Veterinary Manual",
    },
    "lumpy_skin_disease": {
        "title": "Possible lumpy skin disease",
        "summary": "A photo cannot confirm lumpy skin disease. Ask a veterinarian to assess visible lumps or other signs.",
        "precautions": [
            "Contact a veterinarian or local animal-health service for assessment.",
            "Ask the veterinarian about safe separation, insect control, and vaccination guidance for your area.",
            "Do not apply medicines or insecticides unless a veterinarian recommends them.",
        ],
        "source_url": "https://www.msdvetmanual.com/integumentary-system/pox-diseases/lumpy-skin-disease-in-cattle",
        "source_name": "MSD Veterinary Manual",
    },
    "mastitis": {
        "title": "Possible mastitis",
        "summary": "Mastitis needs veterinary assessment. A photo cannot determine its cause or treatment.",
        "precautions": [
            "Contact a veterinarian, especially if the udder is painful or swollen, or the animal seems unwell.",
            "Keep milking and udder-handling equipment clean.",
            "Do not start antibiotics yourself; ask the veterinarian what to do with the milk.",
        ],
        "source_url": "https://www.msdvetmanual.com/reproductive-system/mastitis-in-large-animals/mastitis-in-cattle",
        "source_name": "MSD Veterinary Manual",
    },
    "ringworm": {
        "title": "Possible ringworm",
        "summary": "Ringworm can spread between animals and people. A veterinarian can confirm the cause of skin changes.",
        "precautions": [
            "Use gloves when handling affected skin and wash hands afterwards.",
            "Avoid sharing grooming tools between affected and healthy animals; clean equipment between uses.",
            "Ask a veterinarian to confirm the condition and recommend control steps.",
        ],
        "source_url": "https://www.msdvetmanual.com/integumentary-system/dermatophytosis/dermatophytosis-in-cattle",
        "source_name": "MSD Veterinary Manual",
    },
}


def connect_database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with connect_database() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS care_guidance (
                class_name TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                precautions_json TEXT NOT NULL,
                source_url TEXT,
                source_name TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_band TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        for class_name, guidance in GUIDANCE.items():
            connection.execute(
                """
                INSERT OR IGNORE INTO care_guidance
                    (class_name, title, summary, precautions_json, source_url, source_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    class_name,
                    guidance["title"],
                    guidance["summary"],
                    json.dumps(guidance["precautions"]),
                    guidance["source_url"],
                    guidance["source_name"],
                ),
            )


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        names = json.load(file)

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    if loaded_model.output_shape[-1] != len(names):
        raise ValueError("Model output count does not match class_names.json.")

    return loaded_model, names


initialize_database()
model, class_names = load_model()


def is_allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_confidence_band(confidence):
    if confidence >= HIGH_CONFIDENCE_LIMIT:
        return "higher"
    if confidence >= LOW_CONFIDENCE_LIMIT:
        return "moderate"
    return "low"


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/detection")
def detection():
    return render_template("detection.html")


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose an image before starting the screening.", "error")
        return redirect(url_for("detection"))

    if not is_allowed_file(uploaded_file.filename):
        flash("Please choose a JPG, JPEG, or PNG image.", "error")
        return redirect(url_for("detection"))

    original_name = secure_filename(uploaded_file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{extension}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        # Check that the uploaded file is a valid image.
        uploaded_file.stream.seek(0)
        with Image.open(uploaded_file.stream) as image:
            image.verify()

        uploaded_file.stream.seek(0)
        with Image.open(uploaded_file.stream) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(saved_path)

        # Prepare the image and ask the trained model for a prediction.
        model_image = tf.keras.utils.load_img(
            str(saved_path),
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
        image_array = tf.keras.utils.img_to_array(model_image)
        image_array = np.expand_dims(image_array, axis=0)

        scores = model.predict(image_array, verbose=0)[0]
        best_index = int(np.argmax(scores))
        predicted_class = class_names[best_index]
        confidence = float(scores[best_index])
        confidence_level = get_confidence_band(confidence)

        # Save prediction history in SQLite.
        with connect_database() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions
                    (image_filename, predicted_class, confidence, confidence_band, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    saved_name,
                    predicted_class,
                    confidence,
                    confidence_level,
                    datetime.now().astimezone().isoformat(timespec="minutes"),
                ),
            )
            prediction_id = cursor.lastrowid

        return redirect(url_for("result", prediction_id=prediction_id))

    except (UnidentifiedImageError, OSError, ValueError):
        saved_path.unlink(missing_ok=True)
        flash("That image could not be opened. Try a valid JPG or PNG photo.", "error")
        return redirect(url_for("detection"))

    except Exception:
        saved_path.unlink(missing_ok=True)
        app.logger.exception("Image screening failed.")
        flash("The image could not be screened. Please check the model files and try again.", "error")
        return redirect(url_for("detection"))



@app.get("/history")
def history():
    with connect_database() as connection:
        predictions = connection.execute("""
            SELECT
                predictions.*,
                care_guidance.title AS condition_title
            FROM predictions
            LEFT JOIN care_guidance
                ON care_guidance.class_name = predictions.predicted_class
            ORDER BY predictions.id DESC
        """).fetchall()

        care_guidance = connection.execute("""
            SELECT *
            FROM care_guidance
            ORDER BY title
        """).fetchall()

    return render_template(
        "history.html",
        predictions=predictions,
        care_guidance=care_guidance,
    )




@app.get("/result/<int:prediction_id>")
def result(prediction_id):
    with connect_database() as connection:
        prediction_row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

        if prediction_row is None:
            flash("That result was not found. Please upload an image again.", "error")
            return redirect(url_for("detection"))

        prediction = dict(prediction_row)

        guidance_row = connection.execute(
            "SELECT * FROM care_guidance WHERE class_name = ?",
            (prediction["predicted_class"],),
        ).fetchone()

    if guidance_row:
        guidance = dict(guidance_row)
        guidance["precautions"] = json.loads(guidance["precautions_json"])
    else:
        guidance = {
            "title": "General guidance",
            "summary": "Please ask a veterinarian to review the result.",
            "precautions": ["Do not make treatment decisions from this model result alone."],
            "source_url": None,
            "source_name": None,
        }

    # Low confidence: do not show disease-specific guidance.
    if prediction["confidence_band"] == "low":
        guidance = {
            "title": "Result is uncertain",
            "summary": "Model confidence is low. Ask a veterinarian to assess the animal.",
            "precautions": [
                "Do not treat or move the animal based only on this model result.",
                "If the animal appears unwell, contact a veterinarian promptly.",
                "A clearer, well-lit photo may help with another screening.",
            ],
            "source_url": None,
            "source_name": None,
        }

    predicted_title = GUIDANCE.get(
        prediction["predicted_class"],
        {},
    ).get(
        "title",
        prediction["predicted_class"].replace("_", " ").title(),
    )

    return render_template(
        "result.html",
        prediction=prediction,
        guidance=guidance,
        predicted_title=predicted_title,
    )


@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def upload_too_large(_error):
    flash("Image is too large. Please choose a photo under 10 MB.", "error")
    return redirect(url_for("detection"))


if __name__ == "__main__":
    app.run(debug=True)