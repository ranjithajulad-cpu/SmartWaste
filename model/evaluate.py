import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.metrics import confusion_matrix, classification_report


# =====================================================
# 1. PROJECT PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "smartwaste_model.keras"
)


# =====================================================
# 2. CLASSES
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
# 3. LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully.")


# =====================================================
# 4. COLLECT ALL IMAGES
# =====================================================

image_paths = []
true_labels = []

for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(".jpg"):

            image_paths.append(
                os.path.join(
                    class_folder,
                    filename
                )
            )

            true_labels.append(
                class_index
            )


print()
print("Total images:", len(image_paths))


# =====================================================
# 5. PREDICT EVERY IMAGE
# =====================================================

predicted_labels = []

for number, image_path in enumerate(image_paths):

    img = load_img(
        image_path,
        target_size=(160, 160)
    )

    img_array = img_to_array(img)

   

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_labels.append(
        predicted_index
    )

    if (number + 1) % 100 == 0:

        print(
            "Tested",
            number + 1,
            "/",
            len(image_paths)
        )


# =====================================================
# 6. CONFUSION MATRIX
# =====================================================

true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=range(len(class_names))
)


print()
print("================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


# =====================================================
# 7. CLASSIFICATION REPORT
# =====================================================

print()
print("================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)


# =====================================================
# 8. CLASS-BY-CLASS ACCURACY
# =====================================================

print()
print("================================")
print("CLASS-BY-CLASS ACCURACY")
print("================================")

for i, class_name in enumerate(class_names):

    total = np.sum(
        true_labels == i
    )

    correct = np.sum(
        (true_labels == i) &
        (predicted_labels == i)
    )

    accuracy = (
        correct / total
    ) * 100

    print(
        f"{class_name}: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


print()
print("================================")
print("EVALUATION FINISHED")
print("================================")