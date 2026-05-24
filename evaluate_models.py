import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import os

# --- 1. DATA PREPARATION ---
# Dummy dataset (Replace with actual validation data in production)
X = np.random.rand(30, 224, 224, 3)
y_true = np.random.randint(0, 4, 30)

# --- 2. CONFIGURATION ---
model_paths = {
    "EfficientNet": "models/efficientnet.h5",
    "Xception": "models/xception.h5",
    "MobileNet": "models/mobilenet.h5"
}

results = []

# --- 3. EVALUATION LOOP ---
for name, path in model_paths.items():
    if os.path.exists(path):
        try:
            # Load the pre-trained model
            model = tf.keras.models.load_model(path, compile=False)

            # Generate predictions
            preds = model.predict(X, verbose=0)
            y_pred = np.argmax(preds, axis=1)

            # Calculate and store metrics
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted"),
                "F1": f1_score(y_true, y_pred, average="weighted")
            })
            print(f"Successfully evaluated: {name}")
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    else:
        print(f"Model file not found: {path}")

# --- 4. EXPORT RESULTS ---
if results:
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("\nResults successfully saved to 'results.csv'.")
else:
    print("\nNo models were found to evaluate.")