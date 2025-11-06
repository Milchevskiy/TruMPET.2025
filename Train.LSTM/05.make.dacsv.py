#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm

npy_dir="output"
output_file="da.csv"


def build_dataset_file(npy_dir, output_file, step=5):
    all_rows = []
    all_classes = set()
    num_features = None
    total_rows = 0

    files = [f for f in os.listdir(npy_dir) if f.endswith('.X.npy')]
    files.sort()

    for idx, x_file in enumerate(tqdm(files, desc="Files processing")):
        # обрабатываем только каждый step-й файл
        if idx % step != 0:
            continue

        base = x_file[:-6]
        y_file = f"{base}.y.npy"

        path_x = os.path.join(npy_dir, x_file)
        path_y = os.path.join(npy_dir, y_file)

        if not os.path.exists(path_y):
            print(f"Error: {y_file} not found")
            continue

        try:
            X = np.load(path_x)
            y = np.load(path_y)
        except Exception as e:
            print(f"Reading error {base}: {e}")
            continue

        if len(X) != len(y):
            print(f"Sizes don't coincide: {base}")
            continue

        if num_features is None:
            num_features = X.shape[1]
        elif X.shape[1] != num_features:
            print(f"Descriptors number in {base} are uneven, missed")
            continue

        for label, row in zip(y, X):
            if label == -100:
                continue
            all_rows.append(f"{int(label)} " + " ".join(f"{x:.6f}" for x in row))
            all_classes.add(int(label))
            total_rows += 1

    with open(output_file, "w") as f:
        f.write(f"{len(all_classes)} {total_rows} {num_features}\n")
        for row in all_rows:
            f.write(row + "\n")

    print(f"\nSaved {total_rows} rows, {len(all_classes)} classes, {num_features} descriptors to {output_file}")


# Example of the call
build_dataset_file(
    npy_dir=npy_dir,
    output_file=output_file,
    step=1   # step can be changed to reduce RAM/CPU consumption while SDA/LDA
)
