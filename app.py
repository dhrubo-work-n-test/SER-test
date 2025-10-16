# app.py

import streamlit as st
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torchaudio
import pandas as pd
import os

# ------------------------------
# Step 1: Load model & processor
# ------------------------------
model_name = "superb/wav2vec2-base-superb-er"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
model.eval()  # set model to evaluation mode

# ------------------------------
# Step 2: Streamlit UI
# ------------------------------
st.title("🍔 Food Feedback Sentiment Analysis")
st.write(
    "Upload a voice review after a customer eats, select the food they ordered, "
    "and get the predicted emotion."
)

# Food menu (example: McDonald's)
food_menu = [
    "Big Mac",
    "McChicken",
    "French Fries",
    "McNuggets",
    "Cheeseburger",
    "McFlurry",
    "Happy Meal",
]
selected_food = st.selectbox("Select the ordered food item:", food_menu)

uploaded_file = st.file_uploader("Upload customer's voice review (.wav)", type=["wav"])

# ------------------------------
# Step 3: Prediction
# ------------------------------
if uploaded_file is not None:
    # Save uploaded audio temporarily
    audio_path = os.path.join("temp_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(audio_path, format="audio/wav")

    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze().numpy()

    # Preprocess for model
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Predict emotion
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]

    st.success(f"🎤 Predicted Emotion: **{predicted_label}**")

    # ------------------------------
    # Step 4: Store result in CSV
    # ------------------------------
    data_file = "data.csv"
    new_data = pd.DataFrame({
        "food_item": [selected_food],
        "emotion": [predicted_label]
    })

    if os.path.exists(data_file):
        existing_data = pd.read_csv(data_file)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    updated_data.to_csv(data_file, index=False)
    st.info(f"Data saved to `{data_file}` for dashboard visualization.")
