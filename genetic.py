from sample import *
import numpy as np
import soundfile as sf
import librosa
import pygad
import pygad.nn
import pygad.gann


def denormalize(value, left_limit, right_limit):
  return value * (right_limit - left_limit) + left_limit


def fitness_func(solution, solution_idx):
  # Decode the solution and apply the modifications
  modified_breaks = decode_solution(solution)

  # Merge the modified breaks
  new_track = merge(modified_breaks)

  # Extract features for the new track and the reference track
  new_track_features = extract_drum_features(new_track)
  ref_track_features = extract_drum_features(reference_track)

  # Calculate the cosine similarity
  similarity = compare_tracks_cosine_similarity(new_track_features, ref_track_features)

  # Return the similarity score as fitness
  return similarity

lowest_low_freq = 1
highest_low_freq = 11049
lowest_high_freq = 11051
highest_high_freq = 22049
lowest_pitch = -24
highest_pitch = 24
#mute
lowest_volume = -100
highest_volume = 4


def decode_solution(solution):
  mixed_audios = []
  for idx, parameters in enumerate(solution):
    og_audio, _ = load_break(idx)
    highest_shift = len(og_audio)
    low, high, pitch, shift, db = parameters
    low_fq = denormalize(low, lowest_low_freq, highest_low_freq)
    high_fq = denormalize(high, lowest_high_freq, highest_high_freq)
    pitch_amount = denormalize(pitch, lowest_pitch, highest_pitch)
    shift_amount = denormalize(shift, 0, highest_shift)
    db_amount = denormalize(db, lowest_volume, highest_volume)
    mixed_audio = mix(og_audio, 44100, (low_fq, high_fq), pitch_amount, shift_amount, db_amount)
    mixed_audios.append(mixed_audio)
  return mixed_audios


# Load drum breaks and reference track
drum_breaks, sr = load_random_dnb_breaks(num_breaks=8)
reference_track = mix_and_merge(drum_breaks)

# Define the parameters for the genetic algorithm
num_generations = 100
num_parents_mating = 4
num_solutions = 8

# Define the initial population
initial_population = np.random.uniform(low=0, high=1, size=(num_solutions, len(drum_breaks), 5))

ga_instance = pygad.GA(
  num_generations=num_generations,
  num_parents_mating=num_parents_mating,
  fitness_func=fitness_func,
  initial_population=initial_population,
  sol_per_pop=num_solutions,
)

# Run the genetic algorithm
ga_instance.run()

# Get the best solution
best_solution, best_solution_fitness = ga_instance.best_solution()
print("Best solution fitness:", best_solution_fitness)

# Decode the best solution and apply the modifications
best_modified_breaks = decode_solution(best_solution)

# Merge the best modified breaks
best_new_track = mix_and_merge(best_modified_breaks)

# Save the best new track
librosa.output.write_wav("best_new_track.wav", best_new_track, sr)