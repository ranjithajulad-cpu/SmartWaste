import os
import tensorflow as tf


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartwaste_model.keras"
)


# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 123


# ==========================================
# LOAD MODEL
# ==========================================

print("=" * 50)
print("SMARTWASTE MODEL CHECK")
print("=" * 50)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded.")
print("Input:", model.input_shape)
print("Output:", model.output_shape)


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(dataset.class_names)


# ==========================================
# EVALUATE MODEL
# ==========================================

print("\n" + "=" * 50)
print("EVALUATING MODEL")
print("=" * 50)

loss, accuracy = model.evaluate(
    dataset,
    verbose=1
)


# ==========================================
# RESULT
# ==========================================

print("\n" + "=" * 50)
print("MODEL RESULT")
print("=" * 50)

print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")

print("=" * 50)