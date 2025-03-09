from pprint import pprint
import numpy as np
import copy
import random
import math
from collections import Counter
from pathlib import Path
from rich.progress import track
from rich import print
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.console import Group

N_SOLUTIONS_TO_SHOW = 15

# Preferred shows for certain positions and the
# importance of matching then. A weight of 10 means your preference for
# this dance being in this position is similar to getting 10 quick changes
def setup_preference_listing(N_dances, index_of_dance_after_intermission):
  ACT_1_STARTING_PREFRENCE = None
  ACT_1_PENULTIMATE_PREFRENCE = "Alyssa"
  ACT_1_ENDING_PREFRENCE = "Company"
  ACT_2_STARTING_PREFRENCE = None
  ACT_2_PENULTIMATE_PREFRENCE = "Jamie"
  ACT_2_ENDING_PREFRENCE = "Arielle"
  ACT_1_ANYWHERE_PREFERENCES = []

  # List of (dance ind to prefer, choreographer name, preference score to add if choreographer is at that index)
  preference_listing = []
  if ACT_1_STARTING_PREFRENCE is not None:
    preference_listing.append((0, ACT_1_STARTING_PREFRENCE, 10))
  if ACT_1_PENULTIMATE_PREFRENCE is not None:
    preference_listing.append((index_of_dance_after_intermission - 2, ACT_1_PENULTIMATE_PREFRENCE, 10))
  if ACT_1_ENDING_PREFRENCE is not None:
    preference_listing.append((index_of_dance_after_intermission - 1, ACT_1_ENDING_PREFRENCE, 10))
  if ACT_2_STARTING_PREFRENCE is not None:
    preference_listing.append((index_of_dance_after_intermission, ACT_2_STARTING_PREFRENCE, 10))
  if ACT_2_PENULTIMATE_PREFRENCE is not None:
    preference_listing.append((N_dances - 2, ACT_2_PENULTIMATE_PREFRENCE, 10))
  for act_1_pref_piece in ACT_1_ANYWHERE_PREFERENCES:
    preference_listing.append(((0, index_of_dance_after_intermission - 1), act_1_pref_piece, 10))
  if ACT_2_ENDING_PREFRENCE is not None:
    preference_listing.append((N_dances - 1, ACT_2_ENDING_PREFRENCE, 10))


  for dance_ind, choreo_name, pref_score in preference_listing:
    if choreo_name not in dance_name_to_ind:
      print(f"Error: {choreo_name} is not a listed choreographer")
      exit(0)
  return [(p[0], dance_name_to_ind[p[1]], p[2]) for p in preference_listing]

with open(Path(__file__).parent / 'test_data' / 'season_15.csv', 'r') as f:
  lines = [l.strip().split(',') for l in f.readlines()]

dance_casts = []
dance_ind_to_name = []
dance_name_to_ind = {}
for dance_ind in range(len(lines[0])):
  dance_name = lines[0][dance_ind].strip()
  if dance_name == "":
    assert False, "Please remove any empty columns from the csv"
  dance_cast = {lines[i][dance_ind] for i in range(1, len(lines))} - {''}
  dance_casts.append(dance_cast)
  dance_ind_to_name.append(dance_name)
  dance_name_to_ind[dance_name] = dance_ind
dance_name_to_ind[None] = -1

N_dances = len(dance_casts)
index_of_dance_after_intermission = (N_dances + 1) // 2

preference_listing = setup_preference_listing(N_dances, index_of_dance_after_intermission)
full_cast = set()
for dance_cast in dance_casts:
  full_cast = full_cast.union(dance_cast)

print(
    "The full cast, make sure that no person is repeated twice due to misspellings or pesudeonyms as it will cause invalid show orders:"
)
print(sorted(list(full_cast)))

n_shared_dancers = {}
for dance_ind, dance_cast in enumerate(dance_casts):
  for other_dance_ind, other_cast in enumerate(dance_casts):
    if dance_ind == other_dance_ind:
      continue
    shared_dancers = len(dance_cast.intersection(other_cast))
    n_shared_dancers[(dance_ind, other_dance_ind)] = shared_dancers


def list_quick_changes(show_order) -> list:
  dancers_with_a_back_to_back = set()
  dancers_with_a_quick_change = set()
  for i in range(0, N_dances - 1):
    if i  != index_of_dance_after_intermission - 1:
      dancers_with_a_back_to_back = dancers_with_a_back_to_back.union(dance_casts[show_order[i]].intersection(dance_casts[show_order[i + 1]]))
  for i in range(0, N_dances - 2):
    if (i < index_of_dance_after_intermission) == (i + 2 < index_of_dance_after_intermission):
      dancers_with_a_quick_change = dancers_with_a_quick_change.union(dance_casts[show_order[i]].intersection(dance_casts[show_order[i + 2]]))
  return list(dancers_with_a_quick_change), list(dancers_with_a_back_to_back)


def fitness(show_order):
  n_quick_changes = 0
  back_to_back_changes = 0
  two_piece_changes = 0
  choreo_locations = [-1 for _ in range(N_dances)]
  choreo_locations[show_order[N_dances - 1]] = N_dances - 1
  for i in range(N_dances - 1):
    choreo_locations[show_order[i]] = i
    if i + 3 < N_dances and (i < index_of_dance_after_intermission) == (i + 3 < index_of_dance_after_intermission):
      two_piece_changes += n_shared_dancers[show_order[i], show_order[i + 3]]
    if i + 2 < N_dances and (i < index_of_dance_after_intermission) == (i + 2 < index_of_dance_after_intermission):
      n_quick_changes += n_shared_dancers[(show_order[i], show_order[i + 2])]
    if i  != index_of_dance_after_intermission - 1:
      back_to_back_changes += n_shared_dancers[(show_order[i], show_order[i + 1])]
  pref_matched = 0
  for dance_ind, choreo_ind, pref_score in preference_listing:
    if isinstance(dance_ind, int):
      if show_order[dance_ind] == choreo_ind:
        pref_matched += pref_score
    elif isinstance(dance_ind, tuple):
      if dance_ind[0] <= choreo_locations[choreo_ind] <= dance_ind[1]:
        pref_matched += pref_score
    else:
      assert False
  return (back_to_back_changes, n_quick_changes, pref_matched, two_piece_changes)


def eval_fitness(fitness_output):
  back_to_back_changes, n_quick_changes, pref_matched, two_piece_changes = fitness_output
  return pref_matched - 20 * back_to_back_changes - n_quick_changes - two_piece_changes * .02


swappable_inds = set(range(N_dances))
for ind, n_preferences in Counter([choreo_ind for _, choreo_ind, _ in preference_listing]).items():
  if n_preferences == 1:
    swappable_inds.remove(ind)
swappable_inds = list(swappable_inds)
def mutate(show_order):
  swap_ind, o_swap_ind = np.random.choice(swappable_inds, size=2, replace=True)
  show_order[swap_ind], show_order[o_swap_ind] = \
  show_order[o_swap_ind], show_order[swap_ind]


def print_sequence(show_order):
  pre_intermission_order = [
      dance_ind_to_name[dance_ind]
      for dance_ind in show_order[:index_of_dance_after_intermission]
  ]
  post_intermission_order = [
      dance_ind_to_name[dance_ind]
      for dance_ind in show_order[index_of_dance_after_intermission:]
  ]
  fitness_score = fitness(show_order)
  print(
      f"Back to back changes: {fitness_score[0]}. Quick changes {fitness_score[1]} Two piece changes {fitness_score[3]} Preference points {fitness_score[2]} Score {eval_fitness(fitness_score)}"
  )
  print(','.join(pre_intermission_order))
  print(','.join(["Intermission"] + post_intermission_order), flush=True)
  quick_change_list, back_to_back_list = list_quick_changes(show_order)
  print("Quick changes: " + ", ".join(quick_change_list))
  if len(back_to_back_list):
    print('Back to Back changes: ' + ", ".join(back_to_back_list))

INITIAL_TEMP = 10.0
END_TEMP = 0.01


def simulated_annealing(current_solution, iterations, initial_temp):
  anneal_rate = (END_TEMP / INITIAL_TEMP) ** (1 / iterations)
  current_fitness = eval_fitness(fitness(current_solution))

  best_solution = current_solution
  best_fitness = current_fitness
  T = initial_temp
  while T > END_TEMP:
    new_solution = copy.copy(current_solution)
    mutate(new_solution)
    new_fitness = eval_fitness(fitness(new_solution))

    delta_fitness = new_fitness - current_fitness
    if delta_fitness > 0 or random.random() < math.exp(delta_fitness / T):
      current_solution = new_solution
      current_fitness = new_fitness
    if current_fitness > best_fitness:
      best_solution = copy.copy(current_solution)
      best_fitness = current_fitness
    T *= anneal_rate
  return best_solution, best_fitness


# Generate random starting set of permuations
def gen_random_show_order():
  order = np.random.permutation(N_dances)
  for dance_ind, choreo_ind, pref_score in preference_listing:
    if isinstance(dance_ind, int):
      pref_loc = np.where(order == choreo_ind)
      pref_loc = pref_loc[0][0] # After calling np.where, the shape is ([x],). Get x from this.
      order[pref_loc], order[dance_ind] = order[dance_ind], order[pref_loc]
  return order

with Progress(refresh_per_second=2) as overall_progress:
  def track_in_progress(iterable, description):
    iterable_as_list = list(iterable)
    cur_task_id = overall_progress.add_task(description=description, total=len(iterable_as_list))
    for i in iterable_as_list:
      yield i
      overall_progress.update(cur_task_id, advance=1)
    overall_progress.stop_task(cur_task_id)
    overall_progress.update(cur_task_id, visible=False)

  overall_task_id = overall_progress.add_task("Finding solutions", total=N_SOLUTIONS_TO_SHOW)
  solutions_seen = 0
  best_score = -100
  while solutions_seen < N_SOLUTIONS_TO_SHOW:
    best_score -= 1 # Make it easier to see another solution on redo
    perms = [
        gen_random_show_order()
        for _ in range(int(100_000))
    ]
    perms_zip = [(p, eval_fitness(fitness(p))) for p in track_in_progress(perms, "Evaluating random perms")]
    # Continue with the most promising ones only
    perms_zip = sorted(perms_zip, key=lambda p: -p[1])[:1000]

    # Improve each of the promising ones
    annealed_perms = []
    for p, _ in track_in_progress(perms_zip, "Improving round 1"):
      annealed_perms.append(simulated_annealing(p, 500, 2))
    perms_zip = sorted(annealed_perms, key=lambda p: -p[1])[:100]
    for i, (p, orig_score) in track_in_progress(enumerate(perms_zip), "Final round of improvements"):
      annealed, score = simulated_annealing(p, 20000, 10)
      if score >= best_score:
        best_score = max(best_score, score)
        solutions_seen += 1
        overall_progress.advance(overall_task_id, 1)
        print_sequence(annealed)
      if solutions_seen > N_SOLUTIONS_TO_SHOW:
        break
print("Done")
