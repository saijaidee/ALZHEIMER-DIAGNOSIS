# Stech: Hybrid Alzheimer Diagnosis (EfficientNetB0 + Xception + MobileNetV3) - Streamlit App
This project is a ready template matching the PPT provided by the user.

## What's included
- `app.py` - Streamlit UI to load an MRI image and run inference + Grad-CAM visualization.
- `train.py` - Training script to fine-tune three models and create a soft-voting ensemble.
- `model.py` - Model creation utilities (EfficientNetB0, Xception, MobileNetV3).
- `gradcam.py` - Grad-CAM helper functions.
- `requirements.txt` - Python dependencies.
- `sample_data/` - Place-holder folder. Download OASIS/ADNI or Kaggle dataset and place here.

## How to use
1. Install dependencies:
   `pip install -r requirements.txt`
2. Download dataset (Kaggle/OASIS/ADNI) and extract into `sample_data/`.
3. Train models:
   `python train.py --data_dir sample_data --epochs 10 --batch_size 16`
4. Run Streamlit app:
   `streamlit run app.py`

## Datasets
- Kaggle (easy start): https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
- OASIS-3 / ADNI (for research-grade data): register at their sites.

## Notes
- This repo uses TensorFlow / Keras by default (matches your slide recommendations).
- The training script includes transfer learning, soft voting, and Grad-CAM generation.
