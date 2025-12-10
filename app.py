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


# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import soundfile as sf
# import tempfile
# import pandas as pd

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Voice to Spectrogram + MFCC Extractor", page_icon="🎙️", layout="centered")

# # --- TITLE ---
# st.title("🎙️ Voice to Spectrogram + MFCC Feature Extractor")

# st.write("Record or upload your voice to visualize its spectrogram and extract MFCC features for analysis or modeling.")

# # --- STEP 1: Record or Upload Audio ---
# st.subheader("Step 1: Record or Upload Audio")

# # Option 1: Upload
# audio_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# # Option 2: Record
# st.info("Alternatively, record directly below (if supported by your browser).")

# try:
#     from st_audiorec import st_audiorec
#     audio_bytes = st_audiorec()
# except Exception:
#     audio_bytes = None

# audio_path = None

# # Save uploaded or recorded audio temporarily
# if audio_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#         tmpfile.write(audio_file.read())
#         audio_path = tmpfile.name

# elif audio_bytes is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#         tmpfile.write(audio_bytes)
#         audio_path = tmpfile.name

# # --- SHOW AUDIO PLAYER ---
# if audio_path:
#     st.audio(audio_path, format="audio/wav")
#     st.markdown("---")

# # --- OPTIONS (show only when audio exists) ---
# if audio_path:
#     st.subheader("Additional Settings")

#     # ⭐ Toggle MFCC count (default 13)
#     n_mfcc = st.slider("Select number of MFCCs", 5, 40, 13)

#     # ⭐ Dropdown for colormap
#     cmap_choice = st.selectbox(
#         "Choose spectrogram colormap",
#         ["magma", "viridis", "plasma", "inferno", "coolwarm"]
#     )

# # --- SHOW BUTTON ONLY IF AUDIO IS THERE ---
# start_processing = False
# if audio_path:
#     start_processing = st.button("✨ Click to See the Features")

# # --- PROCESS ONLY IF BUTTON CLICKED ---
# if audio_path and start_processing:

#     with st.spinner("🔄 Processing your audio... Please wait..."):

#         # --- SAFE LOADING ---
#         y, sr = sf.read(audio_path)
#         if y.ndim > 1:
#             y = np.mean(y, axis=1)  # convert stereo to mono

#         # --- Waveform ---
#         st.markdown("## 🎵 Waveform")
#         fig, ax = plt.subplots(figsize=(10, 3))
#         librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
#         ax.set_title("Audio Waveform")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Amplitude")
#         st.pyplot(fig)

#         # --- Mel Spectrogram ---
#         st.markdown("## 🌈 Mel Spectrogram (Log Scale)")
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#         S_dB = librosa.power_to_db(S, ref=np.max)

#         fig, ax = plt.subplots(figsize=(10, 4))
#         img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
#                                        cmap=cmap_choice, ax=ax)
#         ax.set_title(f"Mel Spectrogram ({cmap_choice} colormap)")
#         fig.colorbar(img, ax=ax, format="%+2.0f dB")
#         st.pyplot(fig)

#         # --- MFCC Extraction ---
#         st.markdown("## 🧠 MFCC Feature Extraction")
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#         mfcc_means = np.mean(mfcc, axis=1)

#         fig, ax = plt.subplots(figsize=(10, 4))
#         img = librosa.display.specshow(mfcc, x_axis='time', sr=sr,
#                                        cmap='coolwarm', ax=ax)
#         ax.set_title(f"MFCCs Over Time ({n_mfcc} coefficients)")
#         fig.colorbar(img, ax=ax)
#         st.pyplot(fig)

#         # MFCC Table
#         st.markdown("### 📊 Mean MFCC Feature Values")
#         mfcc_table = pd.DataFrame({
#             f"MFCC {i+1}": [round(mean, 4)] for i, mean in enumerate(mfcc_means)
#         }).T
#         st.table(mfcc_table)

#         # ⭐ Download MFCC as CSV
#         csv_data = pd.DataFrame({"MFCC Index": np.arange(1, n_mfcc+1),
#                                  "Mean Value": mfcc_means})

#         csv_bytes = csv_data.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="📥 Download MFCC as CSV",
#             data=csv_bytes,
#             file_name="mfcc_features.csv",
#             mime="text/csv"
#         )

#     st.success("🎉 Processing complete!")

#     # Reset button
#     if st.button("🔄 Record / Upload Again"):
#         st.experimental_rerun()

# else:
#     if not audio_path:
#         st.warning("Please upload or record an audio file to continue.")
