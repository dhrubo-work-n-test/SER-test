import streamlit as st
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# -------------------------------
# 1️⃣ Page config
# -------------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

st.title("🎤 Speech Emotion Recognition (SER)")
st.write("Upload a WAV file or record your voice to detect the emotion.")

# -------------------------------
# 2️⃣ Load models
# -------------------------------
MODEL_PATH = "SER_Wav2Vec2_Model.h5"
classifier = load_model(MODEL_PATH)

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  # CPU

# -------------------------------
# 3️⃣ Functions
# -------------------------------
def extract_wav2vec2_embeddings(y, sr=16000):
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).numpy()
    return emb

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    st.pyplot(fig)

def predict_emotion(y, sr):
    emb = extract_wav2vec2_embeddings(y, sr)
    preds = classifier.predict(emb)
    pred_idx = np.argmax(preds)
    return emotions[pred_idx], {emotions[i]: float(preds[0][i]) for i in range(len(emotions))}

# -------------------------------
# 4️⃣ Audio input
# -------------------------------
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file is not None:
    y, sr = librosa.load(audio_file, sr=None)
    st.audio(audio_file, format="audio/wav")
    pred_emotion, pred_probs = predict_emotion(y, sr)
    
    st.subheader("Predicted Emotion")
    st.write(f"🎯 {pred_emotion}")

    st.subheader("Class Probabilities")
    st.bar_chart(pred_probs)

    st.subheader("Waveform")
    plot_waveform(y, sr)

st.info("⚡ Note: Model uses Wav2Vec2 embeddings. CPU-only prediction may take a few seconds.")
