import os
import uuid
import logging
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartwaste_model.keras"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================================================
# 2. FLASK APP
# =====================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# Limit uploads to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)


# =====================================================
# 3. LOAD MODEL
# =====================================================

print("========================================", flush=True)
print("SMARTWASTE AI", flush=True)
print("Model path:", MODEL_PATH, flush=True)
print("Model file exists:", os.path.exists(MODEL_PATH), flush=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.", flush=True)
print("Model input shape:", model.input_shape, flush=True)
print("Model output shape:", model.output_shape, flush=True)


# =====================================================
# 4. WASTE CLASSES
# =====================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# =====================================================
# 5. DISPOSAL TIPS
# =====================================================

disposal_tips = {
    "cardboard": (
        "Keep cardboard clean and dry, then place it "
        "with recyclable paper and cardboard."
    ),
    "glass": (
        "Separate glass items and place them in the "
        "appropriate glass recycling container."
    ),
    "metal": (
        "Clean metal containers and place them in the "
        "appropriate recycling collection."
    ),
    "paper": (
        "Keep paper clean and dry and place it with "
        "recyclable paper."
    ),
    "plastic": (
        "Empty and clean recyclable plastic containers "
        "before placing them in plastic recycling."
    ),
    "trash": (
        "Place non-recyclable waste in the appropriate "
        "general-waste bin."
    )
}


# =====================================================
# 6. SERVE UPLOADED IMAGES
# =====================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# =====================================================
# 7. MAIN PAGE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    confidence_level = None
    confidence_message = None
    disposal_tip = None
    image_path = None
    error = None

    if request.method == "POST":

        # ---------------------------------------------
        # CHECK UPLOADED FILE
        # ---------------------------------------------

        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                confidence_level=confidence_level,
                confidence_message=confidence_message,
                disposal_tip=disposal_tip,
                image_path=image_path,
                error=error
            )

        # ---------------------------------------------
        # VALIDATE FILE EXTENSION
        # ---------------------------------------------

        safe_filename = secure_filename(file.filename)
        extension = os.path.splitext(safe_filename)[1].lower()

        allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        if not safe_filename or extension not in allowed_extensions:
            error = "Please upload a JPG, JPEG, PNG, or WEBP image."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                confidence_level=confidence_level,
                confidence_message=confidence_message,
                disposal_tip=disposal_tip,
                image_path=image_path,
                error=error
            )

        # ---------------------------------------------
        # SAVE IMAGE WITH A UNIQUE FILENAME
        # ---------------------------------------------

        image_filename = f"{uuid.uuid4().hex}{extension}"
        saved_image_path = os.path.join(
            UPLOAD_FOLDER,
            image_filename
        )

        try:
            file.save(saved_image_path)

            print("========================================", flush=True)
            print("New image received:", image_filename, flush=True)
            print("Loading image...", flush=True)

            # -----------------------------------------
            # LOAD AND PREPARE IMAGE
            # -----------------------------------------

            img = load_img(
                saved_image_path,
                target_size=(160, 160),
                color_mode="rgb"
            )

            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            print("Image prepared.", flush=True)
            print("Starting model prediction...", flush=True)

            # -----------------------------------------
            # PREDICT WASTE CLASS
            # -----------------------------------------

            predictions = model.predict(
                img_array,
                verbose=0
            )

            print("Prediction completed.", flush=True)

            score = predictions[0]

            if len(score) != len(class_names):
                raise ValueError(
                    f"Expected {len(class_names)} outputs, "
                    f"but received {len(score)}."
                )

            predicted_index = int(np.argmax(score))
            prediction = class_names[predicted_index]

            confidence = round(
                float(score[predicted_index]) * 100,
                2
            )

            # -----------------------------------------
            # CONFIDENCE LEVEL
            # -----------------------------------------

            if confidence >= 70:
                confidence_level = "High confidence"
                confidence_message = (
                    "The model is highly confident in this prediction."
                )

            elif confidence >= 40:
                confidence_level = "Moderate confidence"
                confidence_message = (
                    "The model is reasonably confident, but a clearer "
                    "image may improve accuracy."
                )

            else:
                confidence_level = "Low confidence"
                confidence_message = (
                    "The model is uncertain about this prediction. "
                    "Try uploading a clearer image."
                )

            # -----------------------------------------
            # DISPOSAL TIP AND IMAGE URL
            # -----------------------------------------

            disposal_tip = disposal_tips.get(
                prediction,
                "Follow your local waste-management guidelines."
            )

            image_path = f"/uploads/{image_filename}"

            print("Prediction:", prediction, flush=True)
            print("Confidence:", confidence, "%", flush=True)
            print("========================================", flush=True)

        except Exception:
            app.logger.exception(
                "Image upload or prediction failed"
            )

            error = (
                "Image analysis failed. Please try another image. "
                "If the problem continues, check the server logs."
            )

    # =================================================
    # 8. DISPLAY RESULT
    # =================================================

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        confidence_level=confidence_level,
        confidence_message=confidence_message,
        disposal_tip=disposal_tip,
        image_path=image_path,
        error=error
    )


# =====================================================
# 9. START SERVER LOCALLY
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )