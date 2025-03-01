import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import pyaudio
import wave
import tempfile
import random

# Load the trained model
try:
    model = tf.keras.models.load_model("model.keras")
except Exception as e:
    model = None
    st.warning(f"Model file not found or cannot be loaded! Error: {e}")

# Compile the model with a new optimizer, since we are reloading the model
if model is not None:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])

st.title("CNN Audio Classification App")
st.write("Upload a .wav file or use the microphone to classify if the speech is slurred or not.")

# Option for file upload
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

# Microphone input handling

# Function to preprocess audio
def preprocess_audio_from_ndarray(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=128)
    mfccs = np.mean(mfccs, axis=1)
    return mfccs


# Initialize features variable
features = None  # Reshape to match CNN input shape

# If a file is uploaded, process the file to extract features
if uploaded_file is not None:
    audio_data, _ = librosa.load(uploaded_file, sr=16000)
    features = preprocess_audio_from_ndarray(audio_data)


# Ensure features are defined before using them for prediction
if features is not None:
    features = features.reshape(1, 16, 8, 1)  # Reshape to match CNN input shape

    # Check if the features are scaled the same as during training
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.flatten().reshape(-1, 1)).reshape(1, 16, 8, 1)

    # Predict using the model
    prediction = model.predict(features_scaled)[0][0]

    st.write("Prediction Score:", prediction)

    if prediction > 0.5:
        st.success("The model predicts: Slurred Speech")
    else:
        st.success("The model predicts: Clear Speech")
else:
    st.warning("Please upload a file or record audio to proceed.")
