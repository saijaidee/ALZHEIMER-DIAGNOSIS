import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, Xception, MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np

# Path to your dataset
DATASET_PATH = r"C:\Users\Vardhan\OneDrive\Desktop\Stech_Alzheimer_Project\sample_data\archive\Combined Dataset"

train_dir = os.path.join(DATASET_PATH, "train")
test_dir = os.path.join(DATASET_PATH, "test")

# Folder → Class mapping
CLASS_MAP = {
    "No Impairment": "CN",
    "Very Mild Impairment": "MCI",
    "Mild Impairment": "MCI",
    "Moderate Impairment": "AD"
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_CLASSES = 3  # CN, MCI, AD

def mapped_label_generator(directory, generator):
    data = generator.flow_from_directory(
        directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False
    )

    images, labels = [], []

    for i in range(len(data)):
        batch = data[i]
        filepaths = data.filepaths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        for img, filepath in zip(batch, filepaths):
            folder_name = os.path.basename(os.path.dirname(filepath))
            mapped = CLASS_MAP.get(folder_name)
            if mapped:
                images.append(img)
                labels.append(mapped)

    label_to_index = {"CN": 0, "MCI": 1, "AD": 2}
    y = tf.keras.utils.to_categorical([label_to_index[l] for l in labels], num_classes=NUM_CLASSES)

    return np.array(images), y


print("🔄 Loading training data...")
train_gen = ImageDataGenerator(rescale=1/255.0)
X_train, y_train = mapped_label_generator(train_dir, train_gen)

print("🔄 Loading testing data...")
test_gen = ImageDataGenerator(rescale=1/255.0)
X_test, y_test = mapped_label_generator(test_dir, test_gen)

print("📊 Training Samples:", X_train.shape, "Labels:", y_train.shape)
print("📊 Testing Samples:", X_test.shape, "Labels:", y_test.shape)


def build_model(base):
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)


models = [
    build_model(EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))),
    build_model(Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))),
    build_model(MobileNetV3Large(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
]

model_names = ["efficientnet.h5", "xception.h5", "mobilenet.h5"]

# Compile all models
for model in models:
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])


# --- TRAIN WITH CHECKPOINT ---
for i, (model, filename) in enumerate(zip(models, model_names)):
    print(f"\n🔵 Training Model {i+1} ({filename}) ...")

    checkpoint = ModelCheckpoint(
        filename, monitor="val_accuracy",
        save_best_only=True, verbose=1
    )

    # Resume training if model file already exists
    if os.path.exists(filename):
        print(f"📌 Previous weights found for {filename}, loading...")
        model.load_weights(filename)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=3,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )

print("\n🎉 TRAINING COMPLETE! All models saved with best accuracy.")
