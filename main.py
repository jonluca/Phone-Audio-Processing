from __future__ import print_function

import os

import librosa

EIGHT_KHZ = 8096
SPEECH_LOW_BAND = 200
SPEECH_UPPER_BAND = 3400

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input/game.wav')

time_series, sample_rate = librosa.load(filename, sr=None)

# resample at 8khz
resampled_time_series = librosa.core.resample(y=time_series, orig_sr=sample_rate, target_sr=EIGHT_KHZ)

librosa.output.write_wav("output/eightkhz_resampled.wav", resampled_time_series, EIGHT_KHZ)

short_time_fourier_transform = librosa.core.stft(y=resampled_time_series)

# Clear out speech bands
# TODO(jonluca) is there a better way of doing this? 
for i in range(SPEECH_LOW_BAND):
    short_time_fourier_transform[i] = 0

for i in range(SPEECH_UPPER_BAND, len(short_time_fourier_transform)):
    short_time_fourier_transform[i] = 0

reconstructed_time_series = librosa.core.istft(short_time_fourier_transform)
librosa.output.write_wav("output/eightkhz_resampled_unfrequencied.wav", reconstructed_time_series, EIGHT_KHZ)
