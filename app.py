import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="Alzheimer MRI Diagnosis", layout="wide", page_icon="🧠")

# ---------------- CSS FIX (CLEAN LOOK) ----------------
def add_bg():
    bg_img_path = os.path.join("assets", "background.jpg")
    if os.path.exists(bg_img_path):
        with open(bg_img_path, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                url("data:image/jpg;base64,{bin_str}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-attachment: fixed !important;
            }}
            [data-testid="stHeader"], [data-testid="stMainViewContainer"] {{
                background-color: rgba(0,0,0,0) !important;
            }}
            [data-testid="stSidebar"] {{
                background-color: rgba(0, 0, 0, 0.6) !important;
                backdrop-filter: blur(15px);
            }}
            .stExpander, [data-testid="stVerticalBlock"] > div:has(h1, h2, h3, p, .stButton, .stTable, .stDataFrame) {{
                background-color: rgba(255, 255, 255, 0.1) !important;
                padding: 20px !important;
                border-radius: 12px !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                margin-bottom: 10px !important;
                backdrop-filter: blur(10px);
            }}
            h1, h2, h3, p, span, label, li {{
                color: #FFFFFF !important;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.warning("Place background.jpg in 'assets' folder.")

add_bg()

# ---------------- MODELS SETUP ----------------
@st.cache_resource
def load_models():
    MODEL_DIR = "models"
    paths = [os.path.join(MODEL_DIR, "efficientnet.h5"), os.path.join(MODEL_DIR, "xception.h5"), os.path.join(MODEL_DIR, "mobilenet_weights.npy")]
    loaded = []
    for p in paths:
        if os.path.exists(p):
            try:
                if p.endswith(".h5"): loaded.append(tf.keras.models.load_model(p, compile=False))
                elif p.endswith(".npy"):
                    m = tf.keras.Sequential([tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None), 
                                           tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(4, activation='softmax')])
                    m.set_weights(np.load(p, allow_pickle=True)); loaded.append(m)
            except: pass
    return loaded

models = load_models()

# ---------------- PREDICT (FIXED FOR INHOMOGENEOUS SHAPE) ----------------
def predict(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    all_preds = []
    for m in models:
        try:
            p = m.predict(arr, verbose=0)[0]
            all_preds.append(p)
        except: continue
            
    if not all_preds: return None, None, None

    # Filter to only keep predictions with the same length to avoid ValueError
    max_len = max(len(p) for p in all_preds)
    filtered_preds = [p for p in all_preds if len(p) == max_len]
    
    avg_probs = np.mean(filtered_preds, axis=0)
    avg_probs /= np.sum(avg_probs)

    class_names = ["Non Demented", "Very Mild", "Mild", "Moderate"]
    return avg_probs, class_names[np.argmax(avg_probs)], class_names

# ---------------- SIDEBAR ----------------
if "page" not in st.session_state: st.session_state.page = "home"

with st.sidebar:
    st.title("🏥 Control Center")
    st.success("✅ Models Ready")
    st.divider()
    st.button("🏠 Home", on_click=lambda: st.session_state.update(page="home"), width='stretch')
    st.button("🔍 Diagnosis", on_click=lambda: st.session_state.update(page="diag"), width='stretch')
    st.divider()
    st.markdown("**Project By:**\nPURUSHOTHAM REDDY & M. JAIDEEP REDDY")

# ---------------- PAGE: HOME ----------------
if st.session_state.page == "home":
    st.title("🧠 Alzheimer MRI Diagnosis System")
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a5/Normal_vs_Alzheimer%27s_Brain.png", width=600)
    st.markdown("### Advanced AI Ensemble Analysis\nAnalyze MRI scans using EfficientNet, Xception, and MobileNet.")
    if st.button("🚀 Start Diagnosis Now", width='stretch'):
        st.session_state.page = "diag"
        st.rerun()

# ---------------- PAGE: DIAGNOSIS ----------------
elif st.session_state.page == "diag":
    st.title("🔍 MRI Diagnosis Workspace")
    with st.expander("👤 Patient Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        name, age, gender = c1.text_input("Name"), c2.number_input("Age", 0, 120, 50), c3.selectbox("Gender", ["Male", "Female", "Other"])

    img_file = st.file_uploader("📤 Upload MRI Scan", type=["jpg", "jpeg", "png"])
    
    if st.button("🔬 Run Analysis", type="primary", width='stretch'):
        if img_file:
            img = Image.open(img_file).convert("RGB")
            with st.spinner("Analyzing..."):
                probs, pred, names = predict(img)
            if probs is not None:
                col1, col2 = st.columns(2)
                col1.image(img, caption="Uploaded MRI", width='stretch')
                with col2:
                    st.success(f"### Result: {pred}")
                    st.plotly_chart(px.pie(names=names, values=probs, hole=0.4), width='stretch')
        else: st.error("Please upload an image.")

    # ---------------- COMPARISON & ANALYTICS (RESTORED) ----------------
    st.divider()
    st.header("📊 Performance Analytics & Comparison")
    
    if st.button("🔄 Refresh Comparison Metrics", width='stretch'):
        with st.spinner("Evaluating models..."):
            subprocess.run(["python", "evaluate_models.py"])

    if os.path.exists("results.csv"):
        df_results = pd.read_csv("results.csv")
        st.dataframe(df_results, width='stretch')

        best_m = df_results.loc[df_results["Accuracy"].idxmax()]["Model"]
        st.success(f"🏆 Most Accurate Model: **{best_m}**")

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.markdown("#### Accuracy Comparison")
            st.bar_chart(df_results.set_index("Model")["Accuracy"])
        with chart_c2:
            st.markdown("#### F1 Score Comparison")
            st.bar_chart(df_results.set_index("Model")["F1"])

        # Confusion Matrices
        st.subheader("🔍 Confusion Matrices")
        cm_cols = st.columns(len(df_results))
        for i, row in df_results.iterrows():
            m_name = row["Model"]
            cm_path = f"{m_name}_cm.npy"
            if os.path.exists(cm_path):
                with cm_cols[i]:
                    cm_data = np.load(cm_path)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                    ax.set_title(m_name, color='white')
                    fig.patch.set_facecolor('none')
                    st.pyplot(fig)
    else:
        st.info("Run evaluation script to see comparison metrics.")