import numpy as np
import librosa

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

if __name__ == "__main__":
    audio_file = "audio.wav"
    features = extract_features(audio_file)
    print("Características extraídas:", features)
