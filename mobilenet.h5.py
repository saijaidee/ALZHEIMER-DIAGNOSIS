import tensorflow as tf
import os

# 1. Create the models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

def save_mobilenet():
    print("Creating MobileNet model structure...")
    
    # Load MobileNetV2 without the top layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Add custom layers for your 4 classes
    # (Non Demented, Very Mild, Mild, Moderate)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 2. Save the model as 'mobilenet.h5' in the 'models' folder
    save_path = os.path.join("models", "mobilenet.h5")
    model.save(save_path)
    print(f"✅ Successfully saved: {save_path}")

if __name__ == "__main__":
    save_mobilenet()