from sample import *
import numpy as np
import soundfile as sf
import librosa

audio_files, sr = load_random_dnb_breaks(3)
mixed_break = mix_and_merge(audio_files)

sf.write('exports/test_merge_2.wav', mixed_break, sr, subtype='PCM_24')
