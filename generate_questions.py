import json
import random
from pathlib import Path

QUESTIONS_PATH = Path("output/depth_10_size_1000000/questions.json")

TRAIN_SIZE = 150
VAL_SIZE = 150
TEST_SIZE = 300
TOTAL = TRAIN_SIZE + VAL_SIZE + TEST_SIZE


def main():
    with QUESTIONS_PATH.open() as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    if len(questions) < TOTAL:
        raise ValueError(f"Not enough questions ({len(questions)}) to sample {TOTAL}.")

    subset = random.sample(questions, TOTAL)

    splits = {
        "phantomwiki_train.json": subset[:TRAIN_SIZE],
        "phantomwiki_val.json": subset[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE],
        "phantomwiki_test.json": subset[TRAIN_SIZE + VAL_SIZE :],
    }

    for filename, data in splits.items():
        out_path = QUESTIONS_PATH.parent / filename
        with out_path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {len(data)} questions to {out_path}")


if __name__ == "__main__":
    main()