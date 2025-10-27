import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Voice to Spectrogram + MFCC Extractor", page_icon="🎙️", layout="centered")

# --- TITLE ---
st.title("🎙️ Voice to Spectrogram + MFCC Feature Extractor")

st.write("Record or upload your voice to visualize its spectrogram and extract MFCC features for analysis or modeling.")

# --- STEP 1: Record or Upload Audio ---
st.subheader("Step 1: Record or Upload Audio")

# Option 1: Upload
audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# Option 2: Record
st.info("Alternatively, record directly below (if supported by your browser).")

try:
    from st_audiorec import st_audiorec
    audio_bytes = st_audiorec()
except Exception:
    audio_bytes = None

audio_path = None

# Save uploaded or recorded audio temporarily
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_file.read())
        audio_path = tmpfile.name

elif audio_bytes is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        audio_path = tmpfile.name

# --- STEP 2: Process after user clicks Continue ---
if audio_path:
    st.audio(audio_path, format="audio/wav")

    if st.button("▶️ Continue to Process"):
        # --- SAFE LOADING (no librosa.load) ---
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # convert stereo to mono

        # --- Step 3: Plot Waveform ---
        st.subheader("🎵 Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # --- Step 4: Mel Spectrogram ---
        st.subheader("🌈 Mel Spectrogram (Log Scale)")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
        ax.set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

        # --- Step 5: MFCC Extraction ---
        st.subheader("🧠 MFCC Feature Extraction")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        
        # MFCC Visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm', ax=ax)
        ax.set_title("MFCCs Over Time")
        fig.colorbar(img, ax=ax)
        st.pyplot(fig)
        
        # Display numeric MFCC means
        st.markdown("### 📊 Mean MFCC Feature Values")
        mfcc_table = {f"MFCC {i+1}": [round(val, 4)] for i, val in enumerate(mfcc_means)}
        st.table(mfcc_table)

        # --- Step 6: Reset Button ---
        if st.button("🔄 Record / Upload Again"):
            st.experimental_rerun()

else:
    st.warning("Please upload or record an audio file to continue.")
