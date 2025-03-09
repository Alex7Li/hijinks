import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import solve

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def int2str(x):
    assert x >= 0
    return ("" if x < len(letters) else int2str(x//len(letters) - 1)) + letters[x % len(letters)]

def gen_sample_preferences(seed, out_folder):
    np.random.seed(seed)
    n_dances = 20
    n_dancers = 230
    dance_letters = [int2str(x) for x in range(n_dances)]
    dancer_ids = [f'A{i}' for i in range(n_dancers)]
    max_choreo_preference_list_size = 40
    choreo_preferences_rows = []
    for i in range(n_dances):
        min_dancers = np.random.randint(8, 16)
        max_dancers = np.random.randint(min_dancers, min_dancers + 8)
        choreo_preferences_row_1 = {
            'Dance': dance_letters[i],
            'Min Dancers': min_dancers,
            'Max Dancers': max_dancers,
            'Practice Time': f"Monday {(i * 3) // 4}PM",
            'Choice': 'First Choice',
        }
        choreo_preferences_row_2 = {
            'Dance': '',
            'Min Dancers': '',
            'Max Dancers': '',
            'Practice Time': '',
            'Choice': 'Second Choice',
        }
        dancer_pref_order = np.random.choice(dancer_ids, max_choreo_preference_list_size)
        n_first_choice = np.random.randint(8, 13)
        n_second_choice = np.random.randint(3 * max_dancers // 2 - n_first_choice,
                                            2 * max_dancers - n_first_choice )
        for j in range(max_choreo_preference_list_size):
            col_name = str(j + 1)
            choreo_preferences_row_1[col_name] = ''
            if j < n_first_choice:
                choreo_preferences_row_1[col_name] = dancer_pref_order[j]
            choreo_preferences_row_2[col_name] = ''
            if j < n_second_choice:
                choreo_preferences_row_2[col_name] = dancer_pref_order[j + n_first_choice]
        choreo_preferences_rows.append(choreo_preferences_row_1)
        choreo_preferences_rows.append(choreo_preferences_row_2)
    dancer_preferences_rows = []
    for i in range(n_dancers):
        max_dances = np.random.randint(1, 3)
        n_ranked = np.random.randint(max_dances + 3, n_dances)
        rankings = list(np.random.choice(dance_letters, size=n_ranked, replace=False))
        dancer_preferences_row = {
            'Dancer ID': dancer_ids[i],
            'Max Dances': max_dances,
            'Audit': 'No',
        }
        for dance_letter in dance_letters:
            try:
                dancer_preferences_row[dance_letter] = 1 + rankings.index(dance_letter)
            except ValueError:
                dancer_preferences_row[dance_letter] = ""
        dancer_preferences_rows.append(dancer_preferences_row)

    choreo_pref_path = out_folder / 'choreo_preferences.csv'
    dancer_pref_path = out_folder / 'dancer_preferences.csv'
    pd.DataFrame(choreo_preferences_rows).to_csv(choreo_pref_path, index=False, sep='\t')
    pd.DataFrame(dancer_preferences_rows).to_csv(dancer_pref_path, index=False, sep='\t')
    return choreo_pref_path, dancer_pref_path

def test_solve(tmp_path):
    choreo_pref_path, dancer_pref_path = gen_sample_preferences(seed=0, out_folder=tmp_path)
    def make_args(solve_type):
        return ['testmatching.py','--dancer_preferences_csv', str(dancer_pref_path), '--choreographer_preferences_csv', str(choreo_pref_path),
                 '--output_folder', str(tmp_path), '--solve_type', solve_type]
    sys.argv = make_args('old')
    solve.main()
    sys.argv = make_args('ea')
    solve.main()

if __name__ == '__main__':
    tmp_path = Path(__file__).parent / 'gen_data'
    os.makedirs(tmp_path, exist_ok=True)
    choreo_pref_path, dancer_pref_path = gen_sample_preferences(seed=0, out_folder=tmp_path)

    def make_args(solve_type):
        return ['testmatching.py', '--dancer_preferences_csv', str(dancer_pref_path),
                '--choreographer_preferences_csv', str(choreo_pref_path),
                '--output_folder', str(tmp_path), '--solve_type', solve_type]

    sys.argv = make_args('old')
    solve.main()
    #
    sys.argv = make_args('ea')
    solve.main()

    # sys.argv = make_args('oneshot')
    # solve.main()
