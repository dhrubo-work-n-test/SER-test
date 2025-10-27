import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Voice to Spectrogram + MFCC Extractor",
    page_icon="🎙️",
    layout="centered"
)

# --- TITLE ---
st.title("🎙️ Voice to Spectrogram + MFCC Feature Extractor")
st.write("Record or upload your voice to visualize its spectrogram and extract MFCC features for analysis or modeling.")

# --- SESSION STATE INIT ---
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# --- STEP 1: Record or Upload Audio ---
st.subheader("Step 1: Record or Upload Audio")

# Option 1: Upload
audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# Option 2: Record (optional)
st.info("Alternatively, record directly below (if supported by your browser).")

try:
    from st_audiorec import st_audiorec
    audio_bytes = st_audiorec()
except Exception:
    audio_bytes = None

# --- SAVE AUDIO TEMPORARILY ---
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_file.read())
        st.session_state.audio_path = tmpfile.name
        st.session_state.processed = False

elif audio_bytes is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        st.session_state.audio_path = tmpfile.name
        st.session_state.processed = False


# --- STEP 2: Audio Preview and Continue ---
if st.session_state.audio_path:
    st.audio(st.session_state.audio_path, format="audio/wav")

    if not st.session_state.processed:
        if st.button("▶️ Continue to Process"):
            st.session_state.processed = True
            st.experimental_rerun()


# --- STEP 3: Process Audio (only after Continue) ---
if st.session_state.processed and st.session_state.audio_path:
    audio_path = st.session_state.audio_path

    # --- SAFE LOADING ---
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # convert stereo to mono

    # --- Waveform ---
    st.subheader("🎵 Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # --- Mel Spectrogram ---
    st.subheader("🌈 Mel Spectrogram (Log Scale)")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

    # --- MFCC Extraction ---
    st.subheader("🧠 MFCC Feature Extraction")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm', ax=ax)
    ax.set_title("MFCCs Over Time")
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

    st.markdown("### 📊 Mean MFCC Feature Values")
    mfcc_table = {f"MFCC {i+1}": [round(val, 4)] for i, val in enumerate(mfcc_means)}
    st.table(mfcc_table)

    # --- Reset Button ---
    if st.button("🔄 Record / Upload Again"):
        st.session_state.audio_path = None
        st.session_state.processed = False
        st.experimental_rerun()

else:
    st.warning("Please upload or record an audio file to continue.")
