import tensorflow as tf
import numpy as np
import cv2
import os

# Model weight paths
MODEL_PATHS = ["efficientnet.h5", "xception.h5", "mobilenet.h5"]
IMG_SIZE = (224, 224)

# Label mapping
label_map = {0: "CN (Healthy)", 1: "MCI (Mild Cognitive Impairment)", 2: "AD (Alzheimer's Disease)"}

# ---- Rebuild Models (same architecture used during training) ----
def build_efficientnet():
    base = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def build_xception():
    base = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

def build_mobilenet():
    base = tf.keras.applications.MobileNetV3Large(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(base.input, out)

# Build & load weights
models = [build_efficientnet(), build_xception(), build_mobilenet()]
for m, w in zip(models, MODEL_PATHS):
    if os.path.exists(w):
        m.load_weights(w)
    else:
        print(f"❌ Missing weight file: {w}")

# ---- Image Processing ----
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---- Ensemble Prediction ----
def ensemble_predict(image_path):
    img = preprocess_image(image_path)

    predictions = [model.predict(img, verbose=0)[0] for model in models]
    avg_pred = np.mean(predictions, axis=0)

    final_class = np.argmax(avg_pred)
    confidence = avg_pred[final_class] * 100

    return label_map[final_class], confidence

# ---- Main ----
if __name__ == "__main__":
    img_path = input("Enter MRI Image Path: ").strip()

    if not os.path.exists(img_path):
        print("❌ Image not found!")
    else:
        label, conf = ensemble_predict(img_path)
        print(f"\n🧠 Predicted: {label}")
        print(f"📊 Confidence: {conf:.2f}%")
