import onnxruntime as ort
import numpy as np

def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✔ ONNX Model Loaded: {model_path}")
        return session
    except Exception as e:
        print(f"❌ Failed loading {model_path}: {e}")
        return None

def predict_onnx(session, img_array):
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: img_array})[0]
    return preds
