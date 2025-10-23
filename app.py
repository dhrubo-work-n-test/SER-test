import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile

# --- UI setup ---
st.title("🎙️ Voice to Spectrogram + MFCC Feature Extractor")

st.write("Record your voice or upload an audio file to visualize the spectrogram and extract MFCC features.")

# --- Step 1: Record or Upload Audio ---
st.subheader("Step 1: Record or Upload Audio")

# Option 1: Upload
audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# Option 2: Record (optional mic input)
st.info("Alternatively, you can use the mic recorder below (if supported).")
try:
    from st_audiorec import st_audiorec
    audio_bytes = st_audiorec()
except Exception:
    audio_bytes = None

audio_path = None

# Handle uploaded or recorded audio
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_file.read())
        audio_path = tmpfile.name

elif audio_bytes is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        audio_path = tmpfile.name

# --- Step 2: Process audio ---
if audio_path:
    st.audio(audio_path, format="audio/wav")

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # --- Step 3: Plot waveform ---
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # --- Step 4: Generate and show Spectrogram ---
    st.subheader("Spectrogram (Log-Mel Scale)")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

    # --- Step 5: Extract MFCC features ---
    st.subheader("MFCC Feature Extraction")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # Plot MFCCs
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm', ax=ax)
    ax.set_title("MFCCs Over Time")
    fig.colorbar(ax.images[0], ax=ax)
    st.pyplot(fig)

    # Display numeric MFCC features
    st.markdown("### 📊 Mean MFCC Feature Values")
    mfcc_table = {f"MFCC {i+1}": [val] for i, val in enumerate(mfcc_means)}
    st.table(mfcc_table)

else:
    st.warning("Please upload or record an audio file to continue.")
