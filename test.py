# from pathlib import Path

# path = Path("data/uca-input.jsonl")

# line = path.read_text(encoding="utf-8").splitlines()[88]

# print(repr(line))
# print()
# print(line[:100])

# print(repr(line[60:80]))
# print(repr(line[68:71]))
# print("start")
# for i, c in enumerate(line):
#     if ord(c) < 32:
#         print("CONTROL CHARACTER:", i, repr(c), ord(c))

# print("done")

import json

with open("data/uca-input.jsonl", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"FAILED LINE: {line_num}")
            print(f"ERROR: {e}")
            print(f"POSITION: {e.pos}")
            print(f"AROUND ERROR: {repr(line[max(0,e.pos-30):e.pos+30])}")
            break