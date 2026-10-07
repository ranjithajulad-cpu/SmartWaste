import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==========================================
# SMARTWASTE AI - IMAGE PREDICTION
# ==========================================

print("=" * 40)
print("SMARTWASTE AI - IMAGE PREDICTION")
print("=" * 40)


# Project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_smartwaste_model.keras"
)


# Classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    sys.exit()

model = load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Classes:", classes)


# ==========================================
# GET IMAGE PATH
# ==========================================

if len(sys.argv) < 2:
    print("\nERROR: Please provide an image path.")
    print("\nExample:")
    print(
        r'python model\predict.py "C:\Users\RANJITHA JULAD\SmartWaste\dataset\metal\metal1.jpg"'
    )
    sys.exit()


image_path = sys.argv[1]

print("\nImage:")
print(image_path)


# ==========================================
# CHECK IMAGE
# ==========================================

if not os.path.exists(image_path):
    print("\nERROR: Image not found!")
    print("Image path:", image_path)
    sys.exit()


# ==========================================
# LOAD IMAGE
# ==========================================

try:

    img = image.load_img(
        image_path,
        target_size=(160, 160)
    )

except Exception as e:

    print("\nERROR: Could not load image.")
    print(e)
    sys.exit()


# ==========================================
# PREPROCESS IMAGE
# ==========================================

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)




# ==========================================
# PREDICT
# ==========================================

print("\nPredicting...")

predictions = model.predict(
    img_array,
    verbose=0
)

predictions = predictions[0]


# ==========================================
# SHOW CLASS PROBABILITIES
# ==========================================

print("\n" + "=" * 40)
print("CLASS PROBABILITIES")
print("=" * 40)

for class_name, probability in zip(classes, predictions):

    print(
        f"{class_name.capitalize():10s}: "
        f"{probability * 100:.2f}%"
    )


# ==========================================
# FINAL PREDICTION
# ==========================================

predicted_index = np.argmax(predictions)

predicted_class = classes[predicted_index]

confidence = predictions[predicted_index] * 100


# ==========================================
# CONFIDENCE LEVEL
# ==========================================

if confidence >= 70:
    confidence_level = "High"

elif confidence >= 40:
    confidence_level = "Moderate"

else:
    confidence_level = "Low"


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 40)
print("FINAL PREDICTION")
print("=" * 40)

print("Class:", predicted_class.capitalize())
print(f"Confidence: {confidence:.2f}%")
print("Confidence level:", confidence_level)

print("=" * 40)