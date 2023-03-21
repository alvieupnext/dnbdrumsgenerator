from sample import *
import numpy as np
import soundfile as sf
import librosa

audio_files, sr = load_random_dnb_breaks(5)
mixed_break = mix_and_merge(audio_files)

spec_centroid, spec_bandwidth, spec_contrast, mfcc, stft = extract_drum_features(mixed_break)

mel_spec = librosa.feature.melspectrogram(y=mixed_break, sr=sr)

# y_harmonic, y_perc = librosa.effects.hpss(mixed_break)

# print(spec_bandwidth.shape)
#
# print(spec_contrast.shape)
#
# print(stft.shape)

sf.write('exports/test_merge5.wav', mixed_break, sr, subtype='PCM_24')

# reinvert = librosa.feature.inverse.mel_to_audio(mel_spec)
#
# sf.write('exports/test_merge_invert.wav', reinvert, sr, subtype='PCM_24')

# sf.write('exports/test_harmonic.wav', y_harmonic, sr, subtype='PCM_24')
# sf.write('exports/test_perc.wav', y_perc, sr, subtype='PCM_24')
