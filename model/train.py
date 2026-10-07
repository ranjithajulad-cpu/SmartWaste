import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

dataset_path = os.path.join(
    BASE_DIR,
    "dataset"
)

model_path = os.path.join(
    BASE_DIR,
    "model",
    "smartwaste_model.keras"
)

best_model_path = os.path.join(
    BASE_DIR,
    "model",
    "best_smartwaste_model.keras"
)


# =====================================================
# 2. SETTINGS
# =====================================================

IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
SEED = 123


# =====================================================
# 3. LOAD DATASET
# =====================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print()
print("================================")
print("CLASSES")
print("================================")
print(class_names)


# =====================================================
# 4. DATA PIPELINE
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# =====================================================
# 5. DATA AUGMENTATION
# =====================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])


# =====================================================
# 6. MOBILE NET V2
# =====================================================

base_model = MobileNetV2(
    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    ),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# =====================================================
# 7. BUILD MODEL
# =====================================================

inputs = keras.Input(
    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)


# =====================================================
# 8. COMPILE
# =====================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# 9. CHECKPOINT
# =====================================================

checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


# =====================================================
# 10. INITIAL TRAINING
# =====================================================

print()
print("================================")
print("STARTING INITIAL TRAINING")
print("================================")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[checkpoint]
)


# =====================================================
# 11. FINE-TUNING
# =====================================================

print()
print("================================")
print("STARTING FINE-TUNING")
print("================================")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[checkpoint]
)


# =====================================================
# 12. LOAD BEST MODEL
# =====================================================

print()
print("================================")
print("LOADING BEST MODEL")
print("================================")

best_model = tf.keras.models.load_model(
    best_model_path
)


# =====================================================
# 13. SAVE BEST MODEL AS MAIN MODEL
# =====================================================

best_model.save(
    model_path
)


# =====================================================
# 14. FINISHED
# =====================================================

print()
print("================================")
print("BEST MODEL SAVED SUCCESSFULLY")
print("================================")

print(
    "Model:",
    model_path
)

print(
    "Best checkpoint:",
    best_model_path
)

print(
    "Classes:",
    class_names
)