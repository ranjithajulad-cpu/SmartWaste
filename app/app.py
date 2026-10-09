import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, send_from_directory

from tensorflow.keras.utils import load_img, img_to_array



# =====================================================
# 1. PROJECT PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print()
print("========================================")
print("SMARTWASTE AI")
print("========================================")


# =====================================================
# 2. FLASK APP
# =====================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    )
)


# =====================================================
# 3. UPLOAD FOLDER
# =====================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =====================================================
# 4. MODEL PATH
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartwaste_model.keras"
)

print()
print("Loading model from:")
print(MODEL_PATH)

print()
print(
    "Model file exists:",
    os.path.exists(MODEL_PATH)
)


# =====================================================
# 5. LOAD MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully.")

print(
    "Model input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# =====================================================
# 6. WASTE CLASSES
# =====================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

print(
    "Classes:",
    class_names
)


# =====================================================
# 7. DISPOSAL TIPS
# =====================================================

disposal_tips = {

    "cardboard":
        "Keep cardboard clean and dry, then place it with recyclable paper and cardboard.",

    "glass":
        "Separate glass items and place them in the appropriate glass recycling container.",

    "metal":
        "Clean metal containers and place them in the appropriate recycling collection.",

    "paper":
        "Keep paper clean and dry and place it with recyclable paper.",

    "plastic":
        "Empty and clean recyclable plastic containers before placing them in plastic recycling.",

    "trash":
        "Place non-recyclable waste in the appropriate general-waste bin."
}


# =====================================================
# 8. SERVE UPLOADED IMAGES
# =====================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# =====================================================
# 9. MAIN PAGE
# =====================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None
    confidence = None
    confidence_level = None
    disposal_tip = None
    image_path = None

    # -------------------------------------------------
    # CHECK FOR UPLOAD
    # -------------------------------------------------

    if request.method == "POST":

        file = request.files.get(
            "image"
        )

        if file is None:

            return render_template(
                "index.html",
                error="Please select an image."
            )

        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select an image."
            )


        # =================================================
        # SAVE IMAGE
        # =================================================

        image_filename = file.filename

        saved_image_path = os.path.join(
            UPLOAD_FOLDER,
            image_filename
        )

        file.save(
            saved_image_path
        )


        print()
        print("========================================")
        print("NEW IMAGE")
        print("========================================")

        print(
            "Image:",
            image_filename
        )


        # =================================================
        # LOAD IMAGE
        # =================================================

        img = load_img(
            saved_image_path,
            target_size=(160, 160)
        )


        # =================================================
        # CONVERT IMAGE TO ARRAY
        # =================================================

        img_array = img_to_array(
            img
        )

        # =================================================
        # ADD BATCH DIMENSION
        # =================================================
        
        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # =================================================
        # MODEL PREDICTION
        # =================================================

        predictions = model.predict(
            img_array,
            verbose=0
        )


        # =================================================
        # GET PROBABILITIES
        # =================================================

        score = predictions[0]


        print()
        print("ALL CLASS PROBABILITIES")
        print("----------------------------------------")


        for i, class_name in enumerate(
            class_names
        ):

            percentage = (
                float(score[i]) * 100
            )

            print(
                f"{class_name}: "
                f"{percentage:.2f}%"
            )


        # =================================================
        # GET PREDICTED CLASS
        # =================================================

        predicted_index = int(
            np.argmax(score)
        )

        prediction = class_names[
            predicted_index
        ]


        # =================================================
        # GET CONFIDENCE
        # =================================================

        confidence = round(
            float(
                score[predicted_index]
            ) * 100,
            2
        )


        # =================================================
        # CONFIDENCE LEVEL
        # =================================================

        if confidence >= 70:

            confidence_level = (
                "High confidence"
            )

        elif confidence >= 40:

            confidence_level = (
                "Moderate confidence"
            )

        else:

            confidence_level = (
                "Low confidence"
            )


        # =================================================
        # DISPOSAL TIP
        # =================================================

        disposal_tip = disposal_tips.get(
            prediction,
            "Dispose of this waste according to your local waste-management guidelines."
        )


        # =================================================
        # IMAGE URL
        # =================================================

        image_path = (
            "/uploads/"
            + image_filename
        )


        # =================================================
        # PRINT FINAL RESULT
        # =================================================

        print()
        print("========================================")
        print("FINAL RESULT")
        print("========================================")

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        print(
            "Level:",
            confidence_level
        )

        print(
            "========================================"
        )


    # =================================================
    # SEND RESULT TO HTML
    # =================================================

    return render_template(
        "index.html",

        prediction=prediction,

        confidence=confidence,

        confidence_level=confidence_level,

        disposal_tip=disposal_tip,

        image_path=image_path
    )


# =====================================================
# 10. START FLASK SERVER
# =====================================================

if name == "main":

    print()
    print("========================================")
    print("STARTING SMARTWASTE AI WEBSITE")
    print("========================================")

    print("Open this URL in your browser:")
    print("http://127.0.0.1:5000")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
)