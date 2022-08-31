from rich.progress import track
try:
    with open('codewords/words_alpha.txt') as f:
        words = [w.strip() for w in f.readlines()]
except FileNotFoundError:
    # Wrong directory maybe?
    with open('words_alpha.txt') as f:
        words = [w.strip() for w in f.readlines()]
        
wordSet = set(words)
# Check for codes C such that hash1(C) is a word and hash2(C) is also a word.
# How ambiguous!
pows26 = [pow(26, i) for i in range(100)]
pows36 = [pow(36, i) for i in range(100)]
offset = ord('a')


def hash1(word: str) -> int:
    """Convert a word, interpreted as a base 36 string, to base 10."""
    code = 0
    for i, c in enumerate(word[::-1]):
        code += (ord(c) - offset + 10) * pows36[i]
    return code

def hash2(word: str) -> int:
    """Convert a word, interpreted as a base 26 string, to base 10."""
    code = 0
    for i, c in enumerate(word[::-1]):
        code += (ord(c) - offset) * pows26[i]
    return code

# Note hash2 isn't actually surjective, we assume no words start with 'a'.
def inv_hash2(hash: int) -> str:
    s = ""
    while hash > 0:
        s += chr(hash % 26 + offset)
        hash //= 26
    return s[::-1]

for w in track(words):
    if len(w) > 2: # lots of short nonsense words in the dict
        hashed = hash1(w)
        out_w = inv_hash2(hashed)
        if out_w in wordSet:
            print(f"{w} and {out_w} both come from the code {hashed}")

# By default, it's a really depressingly unintersting set,
# there are only ~3 real/common english pairs:
# nuts and cling
# red and cant
# rio and cats
