import sounddevice as sd
import numpy as np
import librosa
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Path to audio files
data_dir = "C:/Users/Dell/Desktop/abhar3"  # Adjust the path as per your folder
speaker_names = ["abhar", "afkar"]  # Names of the speakers

# Load audio samples and labels
X = []  # Features
y = []  # Labels
for speaker in speaker_names:
    speaker_dir = os.path.join(data_dir, speaker)
    for file in os.listdir(speaker_dir):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join(speaker_dir, file)
            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0)  # Average across time
            X.append(mfccs_processed)
            y.append(speaker)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
model = SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# Save the model
joblib.dump(model, 'speaker_model.pkl')

# Function to recognize the speaker
def recognize_speaker():
    print("Listening... Press Ctrl + C to stop.")
    while True:
        try:
            duration = 5  # Seconds
            audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
            sd.wait()  # Wait until the recording is finished
            audio = audio.flatten()
            mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)

            # Predict the speaker
            prediction = model.predict(mfccs_processed)
            speaker_name = label_encoder.inverse_transform(prediction)
            print(f"Recognized speaker: {speaker_name[0]}")
        
        except KeyboardInterrupt:
            print("Microphone closed. Exiting the program.")
            break

if __name__ == "__main__":
    recognize_speaker()
