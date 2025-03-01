# Eusarthria - Dysarthria-Speech-Refinement App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![gTTS](https://img.shields.io/badge/gTTS-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Torch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-1E88E5?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![SoundDevice](https://img.shields.io/badge/SoundDevice-009688?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD43B?style=for-the-badge&logo=huggingface&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Huggingface-FFDC60?style=for-the-badge&logo=huggingface&logoColor=white)

Check out our Jupyter notebooks for detailed analysis
## Overview
**Eusarthria** is a Streamlit-based web application designed to perform various speech-related tasks, including:
- **Slurred Speech Detection** 96.7% Accuracy on detecting Dysarthria audio - Classification_Model.ipynb
- **Audio To Text** Ensemble learning using Latent Hiearchy and X Hierarchy to refine audio files and AlexNet to caption : Dysarthria_audio_to_text.ipynb
- **LLM Refiner** Prompt Engineered and Fine-Tuned Mistral-7B - **LLM Text Refinement.ipynb**
- **Natural Voice AI Generator** Generated Speech
- **Gradient Accumulation** used for GPU - RAM efficency when training and running models efficently 

### Add Ons -
- **Sentiment Analysis** of transcribed text.
- **Emotion Detection** from speech.
- **Audio Frequency Spectrum Visualization**.
- **Real-time Microphone Recording and Processing**.

## Features of the Streamlit App

### 1. Text-to-Speech (TTS)
- Converts input text to speech.
- Generates an audio file in MP3/Wav format.
- Provides an embedded audio player to listen to the generated speech.
- Visualizes the frequency spectrum of the generated audio.

### 2. Speech-to-Text (STT)
- Works with both uploaded audio files and real-time microphone input.
- Displays transcriptions along with frequency spectrum visualization.
### 3. Frequency Spectrum Analysis
- Demonstrates how audio signals are transformed into frequency spectrums.
- Provides visualization examples for different speech patterns.
    <img width="754" alt="Screenshot 2025-03-01 at 11 10 00 PM" src="https://github.com/user-attachments/assets/ac5465cb-a5b5-429f-a173-2394c1b01e3f" />


### 3. Sentiment Analysis
- Uses a **pre-trained sentiment analysis model** to classify transcribed text as positive, negative, or neutral.

### 4. Emotion Detection
- Predicts emotional tone from speech using an **emotion recognition model**.
- Displays the most probable emotion label along with a confidence score.
  <img width="810" alt="Screenshot 2025-03-01 at 11 18 30 PM" src="https://github.com/user-attachments/assets/7b4c316c-58c3-494d-8510-57ebe337fe03" />


### 5. Audio Classification (Slurred Speech Detection)
- Accepts uploaded `.wav` files for classification.
- Uses a **CNN-based deep learning model** trained to detect slurred speech 
- Outputs whether the speech is clear or slurred based on model predictions.
  <img width="817" alt="Screenshot 2025-03-01 at 11 12 32 PM" src="https://github.com/user-attachments/assets/debe1c59-e9b9-45bc-9f79-3a1515e23701" />



## Dependencies
The project requires the following libraries:
- `streamlit`
- `gtts`
- `torch`
- `librosa`
- `numpy`
- `matplotlib`
- `sounddevice`
- `transformers`
- `tensorflow`
- `scikit-learn`

## How to Run the App

### Prerequisites
Make sure you have **Python 3.7+** installed along with the required dependencies.

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/speech-processing-app.git
   cd speech-processing-app
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the App
To launch the Streamlit app, run:
```sh
streamlit run app.py
```

## Model Training & Dataset
- We first loaded, prepared, and labeled our dataset, identifying high-quality audio samples for model training.
- Subsequently, we harnessed **Wav2Vec2**, **CNN-based classifiers**, and **emotion detection models** to analyze speech patterns effectively.

## Future Enhancements
- Improve **real-time speech recognition** by optimizing microphone input processing.
- Expand **emotion recognition** with a broader dataset.
- Develop a **mobile-friendly UI** for better accessibility.

---
### Contributors
- [Murugappan Venkatesh](https://github.com/Murugapz)
- [Rashi Ojha](https://github.com/Rashi0192)
- [Rishika Mehta](https://github.com/Oganesson0221)
- [Shireen Verma](https://github.com/s812v)
- [Stanley Benjamin Yukon](https://github.com/WinAmazingPrizes)

For any issues or feature requests, please open an issue in the repository.

---
Thank you for using the Eusarthria! 🚀
