import numpy as np
import tensorflow as tf
import os

# Create the folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

def create_npy_weights():
    print("Building MobileNet structure...")
    # Create the exact architecture used in your Alzheimer's project
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax') # 4 Alzheimer's Classes
    ])
    
    # Extract weights as a numpy array
    weights = np.array(model.get_weights(), dtype=object)
    
    # Save the file
    save_path = "models/mobilenet_weights.npy"
    np.save(save_path, weights)
    print(f"✅ Created: {save_path}")

if __name__ == "__main__":
    create_npy_weights()