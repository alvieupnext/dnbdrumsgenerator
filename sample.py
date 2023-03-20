import librosa
import os
import random
import numpy as np
from scipy import signal

def load_random_dnb_breaks(num_breaks=3, break_folder='breaks'):
  """
  Load a specified number of random drum and bass breaks from a folder using Librosa.

  Parameters:
      num_breaks (int): The number of breaks to load. Default is 3.
      break_folder (str): The name of the folder containing the breaks. Default is 'break'.

  Returns:
      break_list (list): A list of NumPy arrays, each representing a loaded break.
  """

  # Get a list of all files in the break folder
  break_files = os.listdir(break_folder)

  # Choose num_breaks files at random
  chosen_files = random.sample(break_files, num_breaks)
  print(f"These breaks were chosen: {chosen_files}")

  # Load the chosen files using Librosa
  break_list = []
  max_length = 0
  for file in chosen_files:
    break_path = os.path.join(break_folder, file)
    dnbbreak, sr = librosa.load(break_path, sr=44100)
    break_list.append(dnbbreak)
    max_length = max(max_length, len(dnbbreak))
  padded_audio = []
  # Generate padding for the break_list
  for brea in break_list:
    padding = max_length - len(brea)
    padded_audio.append(np.pad(brea, (0, padding), mode='constant'))


  return padded_audio, sr

def mix_and_merge(audios):
  sr = 44100
  mixed_breaks = []
  for audio in audios:

    # Choose a random high-pass filter cutoff frequency between 500 and 2000 Hz
    nyquist_freq = sr / 2
    min_cutoff_freq = 500 / nyquist_freq
    max_cutoff_freq = 2000 / nyquist_freq
    cutoff_freq = random.uniform(min_cutoff_freq, max_cutoff_freq)
    # Apply a high-pass filter to isolate high frequencies
    b, a = signal.butter(3, cutoff_freq, 'highpass')
    break_hp = signal.filtfilt(b, a, audio)
    # break_hp = audio

    # Apply random pitch-shifting
    pitch_amount = random.randint(-4, 4)
    pitched_break = librosa.effects.pitch_shift(break_hp, sr=sr, n_steps=pitch_amount)

    # Apply time shifting with a random amount of samples
    shift_amount = random.randint(0, sr // 64)
    filtered_audio = np.roll(pitched_break, shift_amount)




    # # Lower volume by 4db
    filtered_audio *= 10 ** (-4 / 20)

    mixed_breaks.append(filtered_audio)


  #merge breaks together
  merged_mixed_break = mixed_breaks[0]

  for mmb in range(1, len(mixed_breaks)):
    merged_mixed_break += mixed_breaks[mmb]

  # Normalize the volume
  mixed_breaks_norm = librosa.util.normalize(merged_mixed_break)

  return mixed_breaks_norm

def extract_drum_features(audio_file, sr=44100):
  # Compute spectral features
  spec_centroid = librosa.feature.spectral_centroid(y=audio_file, sr=sr)
  spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_file, sr=sr)
  spec_contrast = librosa.feature.spectral_contrast(y=audio_file, sr=sr)
  mfcc = librosa.feature.mfcc(y=audio_file, sr=sr)
  stft = librosa.feature.chroma_stft(y=audio_file, sr=sr)
  return spec_centroid, spec_bandwidth, spec_contrast, mfcc, stft

