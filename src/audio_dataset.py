import numpy as np
import librosa
from dataset import Dataset
import sounddevice as sd
import os
import glob

class AudioDataset(Dataset):

    def _data_points_load(self) -> List[str]:
        pattern_wav = os.path.join(self._root, '**', '*.wav')
        data_points_wav = glob.glob(pattern_wav, recursive=True)
        return data_points_wav

    def _data_load(self, path: str) -> np.ndarray:
        audio, sr = librosa.load(path, sr=None)
        return audio

    def play_audio(self, index: int) -> None:
        data, label = self.__getitem__(index)
        if not self._eager:
            data, sr = librosa.load(data, sr=None)

        print(f"Playing audio. Label: {label if label 
              is not None else 'Label not available'}")
        sd.play(data, sr) 
        sd.wait() 