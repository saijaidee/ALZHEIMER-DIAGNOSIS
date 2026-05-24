import tensorflow as tf
import streamlit as st
import numpy as np

def load_models(model_paths):
    loaded = []
    for path in model_paths:
        try:
            if "mobilenet" in path.lower():
                model = tf.keras.models.load_model(
                    path,
                    custom_objects={"relu6": tf.nn.relu6}
                )
            else:
                model = tf.keras.models.load_model(path)

            st.success(f"✔ Loaded model: {path}")
            loaded.append(model)

        except Exception as e:
            st.error(f"❌ Failed loading {path}: {e}")

    return loaded


def predict_model(model, img_batch):
    """Returns probability predictions from an already loaded model"""
    try:
        pred = model.predict(img_batch)
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.array([0])
