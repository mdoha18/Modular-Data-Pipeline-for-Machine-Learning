from preprocessing import PreprocessingTechnique
import librosa
import numpy as np


class Melspectrogram(PreprocessingTechnique):

    def __init__(self, sampling_rate: int, n_mels: int):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sampling_rate, n_mels=self.n_mels)
        return librosa.power_to_db(mel_spec, ref=np.max)
