import itertools
import copy
import sys
import argparse
from pathlib import Path
from typing import Literal
from rich.progress import Progress, TextColumn
import pandas as pd
import numpy as np

def read_input(dancer_preferences_csv_path, choreographer_preferences_csv_path):
    dancer_preferences_df = pd.read_csv(dancer_preferences_csv_path, delimiter='\t')
    choreo_preferences_df = pd.read_csv(choreographer_preferences_csv_path, delimiter='\t')
    choreo_prefs = {}
    for i in range(0, len(choreo_preferences_df), 2):
        row1 = choreo_preferences_df.iloc[i]
        row2 = choreo_preferences_df.iloc[i + 1]
        assert row1['Choice'] == 'First Choice'
        assert row2['Choice'] == 'Second Choice'
        first_choice = []
        second_choice = []
        for i in itertools.count(1):
            if str(i) not in row1:
                break
            d1 = row1[str(i)]
            d2 = row2[str(i)]
            if not pd.isna(d1) and d1 != "":
                first_choice.append(d1)
            if not pd.isna(d2) and d2 != "":
                second_choice.append(d2)
        first_choice_reverse_dict = {v: i for i, v in enumerate(first_choice)}
        second_choice_reverse_dict = {v: i for i, v in enumerate(second_choice)}
        order_set = set(first_choice + second_choice)
        choreo_prefs[row1['Dance']] = {
            'minDancers': int(row1['Min Dancers']),
            'maxDancers': int(row1['Max Dancers']),
            'practiceTime': row1['Practice Time'],
            'firstChoice': first_choice,
            'secondChoice': second_choice,
            'firstChoiceReverseDict': first_choice_reverse_dict,
            'secondChoiceReverseDict': second_choice_reverse_dict,
            'order': first_choice + second_choice,
            'orderSet': order_set,
        }
    for k in choreo_prefs.keys():
        assert k in dancer_preferences_df.columns, "Choreographer preferences has extra dances"

    dancer_prefs = {}
    for i in range(len(dancer_preferences_df)):
        row = dancer_preferences_df.iloc[i]
        this_dancer_preferences = {
            'maxDances': int(row['Max Dances']),
            'audit': row['Audit'],
        }
        all_prefs = []
        for dance_letter in choreo_prefs.keys():
            pref = row[dance_letter]
            if pd.isna(pref) or pref == "":
                this_dancer_preferences[dance_letter] = float('inf')
            else:
                if int(pref) > 0:
                    all_prefs.append(int(pref))
                this_dancer_preferences[dance_letter] = row[dance_letter]
        if 0 in all_prefs:
            all_prefs.remove(0)
        assert set(all_prefs) == set(range(1, max(all_prefs) + 1)), f"Dancer {row['Dancer ID']} does preferences are not numbered correctly (Missing number?)."
        assert len(all_prefs) == max(all_prefs), f"Dancer {row['Dancer ID']} preferences are not numbered correctly (Duplicate number?)."
        dancer_prefs[row['Dancer ID']] = this_dancer_preferences

    return choreo_prefs, dancer_prefs

def cleanup_preferences(choreo_prefs, dancer_prefs):
    # Remove dancers from choreo lists if they did not rank that piece
    choreo_prefs = copy.deepcopy(choreo_prefs)
    dancer_prefs = copy.deepcopy(dancer_prefs)
    for dance_letter in choreo_prefs.keys():
        for dancer_id in choreo_prefs[dance_letter]['order']:
            if dancer_prefs[dancer_id][dance_letter] == float('inf'):
                choreo_prefs[dance_letter]['order'].remove(dancer_id)
    # Remove dancer preference for a piece if they were not listed in that piece
    for dancer_id in dancer_prefs.keys():
        for dance_letter in choreo_prefs.keys():
                if (dancer_prefs[dancer_id][dance_letter] != float('inf') and dancer_prefs[dancer_id][dance_letter] != 0
                    and dancer_id not in choreo_prefs[dance_letter]['order']):
                    for dance_letter2 in choreo_prefs.keys():
                        if dancer_prefs[dancer_id][dance_letter2] > dancer_prefs[dancer_id][dance_letter]:
                            dancer_prefs[dancer_id][dance_letter2] -= 1
                    dancer_prefs[dancer_id][dance_letter] = float('inf')
    return choreo_prefs, dancer_prefs

def solve_standard(choreo_prefs, dancer_prefs):
    """
    The method previously used for assignment. It has the property
    that adding more preferences will never reduce you chances of
    getting earlier preferences, and if you like a dance and are
    listed high, then you will get in.
    """
    choreo_prefs, dancer_prefs = cleanup_preferences(choreo_prefs, dancer_prefs)

    did_update = True
    def assign(dancer_id, dance_letter, is_in_dance: bool):
        nonlocal did_update, choreo_prefs, dancer_prefs
        did_update = True
        for dance_letter_2 in choreo_prefs.keys():
            if dancer_prefs[dancer_id][dance_letter_2] > dancer_prefs[dancer_id][dance_letter]:
                dancer_prefs[dancer_id][dance_letter_2] -= 1
        dancer_prefs[dancer_id][dance_letter] = 0 if is_in_dance else float('inf')
        if dancer_id in choreo_prefs[dance_letter]['order']:
            choreo_prefs[dance_letter]['order'].remove(dancer_id)
        if is_in_dance:
            choreo_prefs[dance_letter]['minDancers'] -= 1
            choreo_prefs[dance_letter]['maxDancers'] -= 1
            if choreo_prefs[dance_letter]['maxDancers'] == 0:
                # This dance is done.
                choreo_prefs[dance_letter]['order'] = []
                # Update all dancer preferences
                for dancer_id_2 in dancer_prefs.keys():
                    if dancer_prefs[dancer_id_2][dance_letter] != 0:
                        for dance_letter_2 in choreo_prefs.keys():
                            if dancer_prefs[dancer_id_2][dance_letter_2] > dancer_prefs[dancer_id_2][dance_letter]:
                                dancer_prefs[dancer_id_2][dance_letter_2] -= 1
                        dancer_prefs[dancer_id_2][dance_letter] = float('inf')
            dancer_prefs[dancer_id]['maxDances'] -= 1
            if dancer_prefs[dancer_id]['maxDances'] == 0:
                # This dancer is done.
                for dance_letter_2 in choreo_prefs.keys():
                    if dancer_prefs[dancer_id][dance_letter_2] != 0:
                        dancer_prefs[dancer_id][dance_letter_2] = float('inf')
                # Update all choreographer preferences
                for dance_meta in choreo_prefs.values():
                    if dancer_id in dance_meta['order']:
                        dance_meta['order'].remove(dancer_id)
            # Mark dance with the other time as impossible
            practice_time = choreo_prefs[dance_letter]['practiceTime']
            for dance_letter_2 in choreo_prefs.keys():
                if choreo_prefs[dance_letter_2]['practiceTime'] == practice_time and \
                    dance_letter_2 != dance_letter and \
                    float('inf') != dancer_prefs[dancer_id][dance_letter_2]  and \
                    dancer_id in choreo_prefs[dance_letter_2]['order']:
                    if 0 == dancer_prefs[dancer_id][dance_letter_2]:
                        print(f"{dancer_id} is in 2 dances with practice at the same time")
                    assign(dancer_id, dance_letter_2, False)


    # Assign any existing 0 values immediately
    for dancer_id in dancer_prefs.keys():
        for dance_letter in choreo_prefs.keys():
                if dancer_prefs[dancer_id][dance_letter] == 0:
                    assign(dancer_id, dance_letter, True)
                    print(f"Force assigning {dancer_id} to {dance_letter}")
    iteration = 0
    while did_update:
        did_update = False
        for check_dance_letter, check_dance_meta in choreo_prefs.items():
            for i, check_dancer_id in zip(range(check_dance_meta['maxDancers']), check_dance_meta['order']):
                if dancer_prefs[check_dancer_id][check_dance_letter] <= dancer_prefs[check_dancer_id]['maxDances']:
                    assign(check_dancer_id, check_dance_letter, True)
        iteration += 1
    dancer_preferences_solved_standerd_df = pd.DataFrame.from_records(dancer_prefs).transpose()
    return dancer_preferences_solved_standerd_df

def eval_choreographer_happiness(choreo_prefs, solution_to_eval, dance_idx):
    choreo_happiness = 0
    n_dancers = 0
    cur_choreo_prefs = choreo_prefs[dance_idx_to_letter[dance_idx]]
    maxDancers = cur_choreo_prefs['maxDancers']
    for dancer_idx in range(N_DANCERS):
        if solution_to_eval[dance_idx][dancer_idx]:
            n_dancers += 1
            if dancer_idx_to_id[dancer_idx] in cur_choreo_prefs['firstChoiceReverseDict']:
                first_idx = cur_choreo_prefs['firstChoiceReverseDict'][dancer_idx_to_id[dancer_idx]]
                # Between 2/3 and 1 depending on location in firstChoice list
                choreo_happiness += 1 - (first_idx  / len(cur_choreo_prefs['firstChoice'])) / 3
            elif dancer_idx_to_id[dancer_idx] in cur_choreo_prefs['secondChoiceReverseDict']:
                second_idx = cur_choreo_prefs['secondChoiceReverseDict'][dancer_idx_to_id[dancer_idx]]
                # Second choice. Between 0 and 1/3 happiness depending on location in secondChoice list
                choreo_happiness += 1.0 / 3 - (second_idx / len(cur_choreo_prefs['secondChoice'])) / 3
            else:
                return -float('inf')
    if n_dancers > maxDancers:
        return -float('inf')
    missing_dancers = max(0, cur_choreo_prefs['minDancers'] - n_dancers)
    return -2 * missing_dancers + choreo_happiness

def eval_dancer_happiness(dancer_prefs, choreo_prefs, solution_to_eval, dancer_idx):
    n_dances = 0
    cur_dancer_prefs  = dancer_prefs[dancer_idx_to_id[dancer_idx]]
    choice_list = []
    MAXLOG = np.log(N_PIECES + 1)
    times_list = []
    for dance_idx in range(N_PIECES):
        pref_level = cur_dancer_prefs[dance_idx_to_letter[dance_idx]]
        if pref_level == 0 and not solution_to_eval[dance_idx][dancer_idx]:
            return -float('inf') # A piece that must be included was not
        if pref_level == float('inf') and solution_to_eval[dance_idx][dancer_idx]:
            return -float('inf') # A piece that was not ranked was included
        if solution_to_eval[dance_idx][dancer_idx]:
            n_dances += 1
            choice_list += [cur_dancer_prefs[dance_idx_to_letter[dance_idx]]]
            times_list.append(choreo_prefs[dance_idx_to_letter[dance_idx]]['practiceTime'])
    missing_pieces = cur_dancer_prefs['maxDances'] - n_dances
    if missing_pieces < 0:
        return -float('inf')
    # Two dances at the same time
    if len(times_list) != len(set(times_list)):
        return -float('inf')
    preference_boost = 0
    for j, rank in enumerate(sorted(choice_list)):
        # Between 0 and 1 per dance
        preference_boost += 1 - (np.log(rank - j + 1) / MAXLOG)
    return preference_boost - (n_dances == 0)

def mutate(rng, cur_solution, choreo_prefs, dancer_prefs, choreo_score, dancer_score):
    new_solution = np.copy(cur_solution)
    mutate_type_seed = rng.random()
    def mutate_random_position():
        # Flip 1 random position
        random_dance_idx = (rng.random(1) * N_PIECES).astype(int)[0]
        random_dance_letter = dance_idx_to_letter[random_dance_idx]
        candidate_dancer_id = []
        for dancer_id in choreo_prefs[dance_idx_to_letter[random_dance_idx]]['order']:
            if 0 < dancer_prefs[dancer_id][random_dance_letter] < float('inf'):
                candidate_dancer_id.append(dancer_id)
        random_dancer_id = rng.choice(candidate_dancer_id, 1)[0]
        return [random_dance_idx], [dancer_id_to_idx[random_dancer_id]]
    if mutate_type_seed <= .4:
        # Flip a square
        dances_to_flip = rng.choice(list(range(N_PIECES)), 2, replace=False)
        dance_A_idx, dance_B_idx = dances_to_flip[0], dances_to_flip[1]
        dance_A, dance_B = dance_idx_to_letter[dance_A_idx], dance_idx_to_letter[dance_B_idx]
        type1_dancers = (cur_solution[dance_A_idx] == 0) & (cur_solution[dance_B_idx] == 1)
        type2_dancers = (cur_solution[dance_A_idx] == 1) & (cur_solution[dance_B_idx] == 0)
        valid_type1_dancers = []
        valid_type2_dancers = []
        for dancer_id in choreo_prefs[dance_A]['orderSet']:
            dancer_idx = dancer_id_to_idx[dancer_id]
            if type1_dancers[dancer_idx]:
                if dancer_prefs[dancer_id][dance_A] < float('inf') and dancer_prefs[dancer_id][dance_B] != 0:
                    valid_type1_dancers.append(dancer_id)
        for dancer_id in choreo_prefs[dance_B]['orderSet']:
            dancer_idx = dancer_id_to_idx[dancer_id]
            if type2_dancers[dancer_idx]:
                if dancer_prefs[dancer_id][dance_B] < float('inf') and dancer_prefs[dancer_id][dance_A] != 0:
                    valid_type2_dancers.append(dancer_id)

        if len(valid_type1_dancers) == 0 or len(valid_type2_dancers) == 0:
            dance_idxs_to_flip, dancer_idxs_to_flip = mutate_random_position()
        else:
            dancer_A_idx = dancer_id_to_idx[rng.choice(valid_type1_dancers)]
            dancer_B_idx = dancer_id_to_idx[rng.choice(valid_type2_dancers)]
            dance_idxs_to_flip = [dance_A_idx, dance_A_idx, dance_B_idx, dance_B_idx]
            dancer_idxs_to_flip = [dancer_A_idx, dancer_B_idx, dancer_B_idx, dancer_A_idx]
    elif mutate_type_seed <= .7:
        # Move a dancer from dance A to dance B
        dances_to_flip = rng.choice(list(range(N_PIECES)), 2, replace=False)
        dance_A_idx, dance_B_idx = dances_to_flip[0], dances_to_flip[1]
        dance_A, dance_B = dance_idx_to_letter[dance_A_idx], dance_idx_to_letter[dance_B_idx]
        can_move_dancer = (cur_solution[dance_A_idx] == 1) & (cur_solution[dance_B_idx] == 0)
        valid_dancers_to_move = []
        for dancer_id in choreo_prefs[dance_B]['orderSet']:
            dancer_idx = dancer_id_to_idx[dancer_id]
            if can_move_dancer[dancer_idx]:
                if dancer_prefs[dancer_id][dance_B] < float('inf') and dancer_prefs[dancer_id][dance_A] != 0:
                    valid_dancers_to_move.append(dancer_id)
        if len(valid_dancers_to_move) == 0:
            dance_idxs_to_flip, dancer_idxs_to_flip = mutate_random_position()
        else:
            dancer_to_move = dancer_id_to_idx[rng.choice(valid_dancers_to_move)]
            # If the dances are at different times and the first dance is not at max dancers,
            # then we don't need to drop the first dance
            dance_idxs_to_flip = [dance_A_idx, dance_B_idx]
            dancer_idxs_to_flip = [dancer_to_move, dancer_to_move]
            if  (
                choreo_prefs[dance_A]['practiceTime'] != choreo_prefs[dance_B]['practiceTime'] and
                choreo_prefs[dance_B]['maxDancers'] > np.sum(cur_solution[dance_B_idx])
            ):
                dance_idxs_to_flip = [dance_B_idx]
                dancer_idxs_to_flip = [dancer_to_move]
    else:
        # Switch 1 member of a dance cast
        dance_to_change_idx = rng.choice(list(range(N_PIECES)))
        dance_to_change = dance_idx_to_letter[dance_to_change_idx]
        dancer_idx_to_remove = []
        dancer_idx_to_add = []
        in_dances = np.sum(cur_solution, axis=0)
        for dancer_id in choreo_prefs[dance_to_change]['orderSet']:
            dancer_idx = dancer_id_to_idx[dancer_id]
            if dancer_prefs[dancer_id][dance_to_change] < float('inf') and in_dances[dancer_idx] < dancer_prefs[dancer_id]['maxDances']:
                dancer_idx_to_add.append(dancer_idx)
            if dancer_prefs[dancer_id][dance_to_change] != 0:
                dancer_idx_to_remove.append(dancer_idx)
        if len(dancer_idx_to_add) == 0:
            dance_idxs_to_flip, dancer_idxs_to_flip = mutate_random_position()
        else:
            dancer_id_to_add = rng.choice(dancer_idx_to_add)
            if len(dancer_idx_to_remove) == 0:
                dancer_idxs_to_flip = [dancer_id_to_add]
            else:
                dancer_id_to_remove = rng.choice(dancer_idx_to_remove)
                dancer_idxs_to_flip = [dancer_id_to_remove, dancer_id_to_add]
            dance_idxs_to_flip = [dance_to_change_idx for _ in dancer_idxs_to_flip]

    new_choreo_score = np.copy(choreo_score)
    new_dancer_score = np.copy(dancer_score)
    for dance_idx_to_flip, dancer_idx_to_flip in zip(dance_idxs_to_flip, dancer_idxs_to_flip):
        new_solution[dance_idx_to_flip, dancer_idx_to_flip] = 1 - cur_solution[dance_idx_to_flip, dancer_idx_to_flip]
    for dance_idx_to_flip in set(dance_idxs_to_flip):
        score = eval_choreographer_happiness(choreo_prefs, new_solution, dance_idx_to_flip)
        if score == -float('inf'):
            return -float('inf'), new_solution, new_choreo_score, new_dancer_score
        new_choreo_score[dance_idx_to_flip] = score
    for dancer_idx_to_flip in set(dancer_idxs_to_flip):
        score = eval_dancer_happiness(dancer_prefs, choreo_prefs, new_solution, dancer_idx_to_flip)
        if score == -float('inf'):
            return -float('inf'), new_solution, new_choreo_score, new_dancer_score
        new_dancer_score[dancer_idx_to_flip] = score
    updated_score = np.sum(new_choreo_score) + np.sum(new_dancer_score)
    return updated_score, new_solution, new_choreo_score, new_dancer_score

def grow_baby(rng, cur_solution, choreo_prefs, dancer_prefs, max_t, max_iters, progress):
    grow_task = progress.add_task("[green]Evolving", total=max_t)
    choreo_score = np.array([eval_choreographer_happiness(choreo_prefs, cur_solution, i) for i in range(N_PIECES)], dtype=np.float32)
    dancer_score = np.array([eval_dancer_happiness(dancer_prefs, choreo_prefs, cur_solution, i) for i in range(N_DANCERS)], dtype=np.float32)
    cur_score = np.sum(choreo_score) + np.sum(dancer_score)
    if cur_score == -float('inf'):
       print("Starting solution wasn't valid, answer may be bad")
    best_score = cur_score
    best_solution = cur_solution
    cur_temp = max_t
    for _ in itertools.count():
        new_score, mutated_solution, mutated_choreo_score, mutated_dancer_score = \
            mutate(rng, cur_solution, choreo_prefs, dancer_prefs, choreo_score, dancer_score)
        if new_score > cur_score or np.exp((new_score - cur_score) / cur_temp) > rng.random():
            cur_score = new_score
            choreo_score = mutated_choreo_score
            dancer_score = mutated_dancer_score
            cur_solution = mutated_solution
            if cur_score > best_score:
                best_score = cur_score
                best_solution = mutated_solution
        cur_temp -= max_t / max_iters
        if cur_temp <= 0:
            break
        progress.update(grow_task, advance=max_t/max_iters, description=f"[green]Best {best_score:.2f} Cur {cur_score:.2f}")
    progress.remove_task(grow_task)
    return best_score, best_solution

def make_baby(rng, daddy, mommy):
    start_solution = daddy & mommy
    pieces_gene = rng.random(N_PIECES)
    dancers_gene = rng.random(N_DANCERS)
    for row in range(N_PIECES):
        for col in range(N_DANCERS):
            if pieces_gene[row] < .5 and dancers_gene[col] < .5:
                start_solution[row][col] = mommy[row][col]
            elif pieces_gene[row] > .5 and dancers_gene[col] > .5:
                start_solution[row][col] = daddy[row][col]
    return start_solution

def solve_oneshot(choreo_prefs, dancer_prefs):
    rng = np.random.default_rng(seed=RANDOM_SEED)
    start_solution = np.zeros((N_PIECES, N_DANCERS), dtype=bool)
    with Progress(*Progress.get_default_columns()) as progress:
        score, solution = grow_baby(rng, start_solution, choreo_prefs, dancer_prefs,
                                    max_t=1.0, max_iters=5 * COMPUTE_BUDGET_PER_GENERATION, progress=progress)
    print(score)
    raise AssertionError("Saving outputs not implemented for oneshot mode")


COMPUTE_BUDGET_PER_GENERATION = 100000

def solve_genetic(choreo_prefs, dancer_prefs, output_folder):
    rng = np.random.default_rng(seed=RANDOM_SEED)
    gen_solutions = [(-float('inf'), np.zeros((N_PIECES, N_DANCERS), dtype=bool)) for _ in range(1)]
    n_genes_to_keep = 1
    best_solution = gen_solutions[0][1]
    best_score = gen_solutions[0][0]
    best_score_each_gen = []
    with Progress(*Progress.get_default_columns()) as progress:
        overall_task = progress.add_task("[red]Gen # 1", total=None)
        for generation in itertools.count(1):
            next_gen_solutions = []
            generation_size = max(20, 60 - 4 * generation)
            generation_task = progress.add_task("[blue]Generation Progress", total=generation_size)
            for _ in range(generation_size):
                parent_inds = rng.choice(range(n_genes_to_keep), 2)
                baby = make_baby(rng, gen_solutions[parent_inds[0]][1], gen_solutions[parent_inds[1]][1])
                next_gen_solutions.append(grow_baby(rng, baby, choreo_prefs, dancer_prefs,
                                                    max_t=0.6, max_iters=COMPUTE_BUDGET_PER_GENERATION // generation_size, progress=progress))
                progress.update(generation_task, advance=1, description=f"[blue]Last score {next_gen_solutions[-1][0]:.2f} Matches {np.sum(next_gen_solutions[-1][1])}")
            progress.remove_task(generation_task)
            # Sort children by how good they are
            next_gen_solutions = sorted(gen_solutions + list(next_gen_solutions), key=lambda x: -x[0])
            # Get the first few children that are not too similar
            n_genes_to_keep = generation_size // 5
            gen_solutions = []
            for score, solution in next_gen_solutions:
                min_d = N_PIECES * N_DANCERS
                for _, solution_2 in gen_solutions:
                    min_d = min(min_d, np.sum(solution ^ solution_2))
                if min_d > N_PIECES * N_DANCERS * .003:
                    gen_solutions.append((score, solution))
                if len(gen_solutions) >= n_genes_to_keep:
                    break
            best_score_each_gen.append(gen_solutions[0][0])
            if gen_solutions[0][0] > best_score:
                best_solution = gen_solutions[0][1]
                best_score = gen_solutions[0][0]
            # print(f"Best Scores Generation {generation} (higher better): {[f'{gen_solutions[i][0]:.2f}' for i in range(len(gen_solutions))]}. Total assignments {np.sum(gen_solutions[0][1])}")
            # 5 Generations with very little improvement
            if len(best_score_each_gen) > 5 and best_score_each_gen[-5] > best_score - .5:
                print(best_score_each_gen, best_score)
                break
            end_solution_df = pd.DataFrame(data=best_solution.T, index=dancer_idx_to_id, columns=dance_idx_to_letter)
            dancer_preferences_solved_df_formatted = format_df(choreo_prefs, dancer_prefs,
                                                               end_solution_df, show_original_rankings=True)
            dancer_preferences_solved_df_formatted.to_csv(output_folder / 'dancer_preferences_solved.csv')
            progress.update(overall_task, advance=1, description=f'[red]Gen # {generation} Best {gen_solutions[0][0]:.2f}')

    end_solution_df = pd.DataFrame(data=best_solution.T, index=dancer_idx_to_id, columns=dance_idx_to_letter)
    return end_solution_df

def format_df(choreo_prefs, dancer_prefs, solved_df, show_original_rankings, print_info=False):
    solution_binary = np.zeros((N_PIECES, N_DANCERS), dtype=bool)
    solved_df = solved_df.copy()
    for dance_letter in choreo_prefs.keys():
        for dancer_id in dancer_prefs.keys():
            solution_binary[dance_letter_to_idx[dance_letter]][dancer_id_to_idx[dancer_id]] = solved_df[dance_letter][dancer_id] == 1

    solved_df['maxDances'] = [dancer_prefs[dancer_id]['maxDances'] for dancer_id in sorted(dancer_prefs.keys())]
    solved_df['audit'] = [dancer_prefs[dancer_id]['audit'] for dancer_id in sorted(dancer_prefs.keys())]
    solved_df['happiness'] = [eval_dancer_happiness(dancer_prefs, choreo_prefs, solution_binary, dancer_id_to_idx[dancer_id]) for dancer_id in sorted(dancer_prefs.keys())]
    final_happiness = sum(solved_df['happiness'])
    meta_df = pd.DataFrame(columns=solved_df.columns)
    for choreo_preference_metadata_column_name in ['maxDancers', 'minDancers', 'practiceTime', 'happiness']:
        row_data = []
        for dance_letter in solved_df.columns:
            if dance_letter in choreo_prefs:
                if choreo_preference_metadata_column_name == 'happiness':
                    choreo_happiness = eval_choreographer_happiness(choreo_prefs, solution_binary, dance_letter_to_idx[dance_letter])
                    final_happiness += choreo_happiness
                    row_data.append(choreo_happiness)
                else:
                    row_data.append(choreo_prefs[dance_letter][choreo_preference_metadata_column_name])
            else:
                row_data.append("")
        meta_df.loc[choreo_preference_metadata_column_name] = row_data
    solved_df = pd.concat([meta_df, solved_df])
    for dance_letter, cur_choreo_preferences in choreo_prefs.items():
        for dancer_id, cur_dancer_preferences in dancer_prefs.items():
            dancer_ranking_of_current_dance = cur_dancer_preferences[dance_letter]
            if dancer_ranking_of_current_dance == float('inf'):
                dancer_ranking_of_current_dance = '-'
            else:
                dancer_ranking_of_current_dance = int(dancer_ranking_of_current_dance)
            choreo_ranking_of_current_dancer = '-'
            if dancer_id in cur_choreo_preferences['firstChoice']:
                choreo_ranking_of_current_dancer = f"1:{cur_choreo_preferences['firstChoice'].index(dancer_id)}"
            if dancer_id in cur_choreo_preferences['secondChoice']:
                choreo_ranking_of_current_dancer = f"2:{cur_choreo_preferences['secondChoice'].index(dancer_id)}"
            if show_original_rankings:
                solved_df[dance_letter][dancer_id] = (
                    f"({dancer_ranking_of_current_dance}, {choreo_ranking_of_current_dancer}) " +
                    f"{int(solution_binary[dance_letter_to_idx[dance_letter]][dancer_id_to_idx[dancer_id]])}")
            else:
                solved_df[dance_letter][dancer_id] = int(solution_binary[dance_letter_to_idx[dance_letter]][dancer_id_to_idx[dancer_id]])
    if print_info:
        print(f"Final happiness {final_happiness}")
        print(f"N matches: {np.sum(solution_binary)}")
    return solved_df.reindex(sorted(solved_df.columns, key=lambda x:(1, x) if x[0].isupper() else (0, x)), axis=1)


dancer_idx_to_id = []
dancer_id_to_idx = {}
dance_idx_to_letter = []
dance_letter_to_idx = {}
N_DANCERS, N_PIECES = 0,0
WARMUP_STEPS = 200
RANDOM_SEED = 0

def main():
    global dancer_idx_to_id, dancer_id_to_idx, dance_idx_to_letter, dance_letter_to_idx, N_DANCERS, N_PIECES
    parser = argparse.ArgumentParser()
    parser.add_argument("--dancer_preferences_csv",
                        type=Path,
                        default=Path(__file__).parent / 'dancer_preferences.csv')
    parser.add_argument("--choreographer_preferences_csv",
                        type=Path,
                        default=Path(__file__).parent / 'choreographer_preferences.csv')
    parser.add_argument("--solve_type",
                        type=str,
                        default='ea')
    parser.add_argument("--output_folder",
                        type=Path,
                        default=Path(__file__).parent)

    args = parser.parse_args()
    choreo_preferences, dancer_preferences = read_input(args.dancer_preferences_csv, args.choreographer_preferences_csv)
    N_DANCERS, N_PIECES = len(dancer_preferences.keys()), len(choreo_preferences.keys())
    dancer_idx_to_id = []
    dancer_id_to_idx = {}
    for i, d in enumerate(sorted(dancer_preferences.keys())):
        dancer_idx_to_id.append(d)
        dancer_id_to_idx[d] = i
    dance_idx_to_letter = []
    dance_letter_to_idx = {}
    for i, l in enumerate(sorted(choreo_preferences.keys())):
        dance_idx_to_letter.append(l)
        dance_letter_to_idx[l] = i

    if args.solve_type == 'old':
        dancer_preferences_solved_df = solve_standard(choreo_preferences, dancer_preferences)
        dancer_preferences_solved_df.replace(list(range(1, 1+len(choreo_preferences))), -1, inplace=True)
        dancer_preferences_solved_df.replace(0, 1, inplace=True)
        dancer_preferences_solved_df.replace(float('inf'), 0, inplace=True)
        dancer_preferences_solved_df.replace(-1, '?', inplace=True)
    elif args.solve_type == 'ea':
        dancer_preferences_solved_df = solve_genetic(choreo_preferences, dancer_preferences, args.output_folder)
    elif args.solve_type == 'oneshot':
        solve_oneshot(choreo_preferences, dancer_preferences)
    else:
        raise argparse.ArgumentError(f'Invalid solve type {args.solve_type}')
    dancer_preferences_solved_df_formatted = format_df(choreo_preferences, dancer_preferences, dancer_preferences_solved_df, show_original_rankings=True, print_info=True)
    dancer_preferences_solved_df_formatted.to_csv(args.output_folder / 'dancer_preferences_solved.csv')
    dancer_preferences_solved_df_formatted_2 = format_df(choreo_preferences, dancer_preferences, dancer_preferences_solved_df, show_original_rankings=False)
    dancer_preferences_solved_df_formatted_2.to_csv(args.output_folder / 'dancer_preferences_solved_simple.csv')

if __name__ == '__main__':
    main()
