import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="🎤 Speech Emotion Recognition", layout="centered")
st.title("🎧 Speech Emotion Recognition Demo")
st.write("Upload a voice sample (.wav) to predict its emotion!")

# -----------------------------
# Load the Model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("SML_SER_Model.h5")
    return model

model = load_model()

# -----------------------------
# Helper Functions
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.flatten()

    # Ensure input length matches training
    expected_len = 2160
    if len(mfcc) < expected_len:
        mfcc = np.pad(mfcc, (0, expected_len - len(mfcc)))
    else:
        mfcc = mfcc[:expected_len]

    # Reshape for model input
    mfcc = np.expand_dims(mfcc, axis=(0, 2))
    return mfcc

def predict_emotion(model, mfcc_input):
    preds = model.predict(mfcc_input)
    pred_idx = np.argmax(preds)
    return pred_idx, preds

# Emotion labels (edit if your dataset differs)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# -----------------------------
# File Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    mfcc_input = extract_features("temp.wav")
    pred_idx, preds = predict_emotion(model, mfcc_input)

    st.subheader(f"🎯 Predicted Emotion: **{emotion_labels[pred_idx].upper()}**")
    st.write("Confidence scores:")
    for i, label in enumerate(emotion_labels):
        st.write(f"- {label.capitalize()}: {preds[0][i]:.4f}")
