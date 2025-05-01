from preprocessing import PreprocessingTechnique
from padding import Padding
from melspec import Melspectrogram
from center_crop import CenterCrop
from pitch_shifting import PitchShifting


class Pipeline(PreprocessingTechnique):
    def __init__(self, *preprocessing_steps):
        self.preprocessing_steps = preprocessing_steps

    def __call__(self, data):
        for step in self.preprocessing_steps:
            data = step(data)
        return data


padding = Padding(target_height=400, target_width=500, color=(255, 192, 203))
melspectrogram = Melspectrogram(sampling_rate=None, n_mels=128)
pitch_shifting = PitchShifting(factor=2, sampling_rate=None)
center_crop = CenterCrop(height=100, width=100)

preprocessing_pipeline = Pipeline (pitch_shifting, melspectrogram)

old_audio_data = "/Users/mekholadoha/Desktop/OOP_FP/project_dataset/audio_dataset/audio_data/disco/71.wav"
new_audio_data = preprocessing_pipeline(old_audio_data)
