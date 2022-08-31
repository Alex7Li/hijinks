# pip install PyQt5 stockfish
from difflib import Match
from stockfish import Stockfish
import os
from pathlib import Path

def setup_engines():
    # Installed from https://stockfishchess.org/download/linux/
    # !wget https://stockfishchess.org/files/stockfish_15_linux_x64.zip
    # !unzip -q stockfish_15_linux_x64.zip
    # !chmod 777 stockfish_15_linux_x64/stockfish_15_x64
    path = Path(".") / "shockfish" / "stockfish_15_linux_x64/stockfish_15_x64"
    weakfish = Stockfish(path, depth=5, parameters={"Slow Mover": 0, "Minimum Thinking Time": 10})
    strongfish = Stockfish(path, depth=15, parameters={"Slow Mover": 0})
    return weakfish, strongfish

def get_reasonable_moves(fish, is_white, n_moves, allowed_centipawn_loss):
    top_moves = fish.get_top_moves(n_moves)
    sign = 1 if is_white else -1
    def score(move_info):
        centipawn = move_info['Centipawn'] * sign
        if centipawn is None:
            mate = move_info['Mate'] * sign
            if mate < 0:
                centipawn = -100_000 + move_info['Mate']
            else:
                centipawn = 100_000 - move_info['Mate']
        return centipawn
    move_scores = [score(move) for move in top_moves]
    best_score = max(move_scores)
    reasonable_moves_with_score = dict(
        (move['Move'], score) for move, score in zip(top_moves, move_scores)
        if score > best_score - allowed_centipawn_loss)
    return reasonable_moves_with_score

def best_move_of(move_dict):
    move_score_zip = list(zip(move_dict.keys(), move_dict.values()))
    return max(move_score_zip, key=lambda m:m[1])[0]

def find_move(weakfish, strongfish, is_white):
    # Even a foolfish could find these reasonable moves
    good_looking = get_reasonable_moves(weakfish, is_white, 30, 80)
    # Good moves
    good_moves = get_reasonable_moves(strongfish, is_white, 10, 40)

    surprising_move_keys = good_moves.keys() - good_looking.keys()
    surprising_moves = dict([(key, good_moves[key]) for key in surprising_move_keys])
    if len(surprising_moves) == 0:
        return best_move_of(good_moves), False
    return best_move_of(surprising_moves), True

if __name__ == '__main__':
    weakfish, strongfish = setup_engines()
    is_white = True
    while(True):
        print(f"{'White' if is_white else 'Black'} to move. {strongfish.get_evaluation()}")
        move, is_shocking = find_move(weakfish, strongfish, is_white)
        weakfish.make_moves_from_current_position([move])
        strongfish.make_moves_from_current_position([move])
        is_white = not is_white
        if(is_shocking):
            print("They'll never see this coming... {move}!")
            print(strongfish.get_board_visual())
            input()
        else:
            print(strongfish.get_board_visual())


