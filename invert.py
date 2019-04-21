from librosa.feature import inverse
import numpy as np
import torch
from scipy.io import wavfile

if __name__ == '__main__':
    mel = np.load('data/mel/p225/p225_013.npy').astype(np.float32)
    audio_arr = inverse.mel_to_audio(mel, sr=48000, n_iter=10)
    wavfile.write('test.wav', 48000, audio_arr)