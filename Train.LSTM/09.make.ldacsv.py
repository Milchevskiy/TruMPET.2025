#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm

def build_dataset_file(npy_dir, output_file, step=5):
    all_rows = []
    all_classes = set()
    num_features = None
    total_rows = 0

    files = [f for f in os.listdir(npy_dir) if f.endswith('.X.npy')]
    files.sort()

    for idx, x_file in enumerate(tqdm(files, desc="Обработка файлов")):
        # обрабатываем только каждый step-й файл
        if idx % step != 0:
            continue

        base = x_file[:-6]
        y_file = f"{base}.y.npy"

        path_x = os.path.join(npy_dir, x_file)
        path_y = os.path.join(npy_dir, y_file)

        if not os.path.exists(path_y):
            print(f"⚠️ Пропущено: {y_file} не найдено")
            continue

        try:
            X = np.load(path_x)
            y = np.load(path_y)
        except Exception as e:
            print(f"❌ Ошибка при чтении {base}: {e}")
            continue

        if len(X) != len(y):
            print(f"⚠️ Размеры не совпадают: {base}")
            continue

        if num_features is None:
            num_features = X.shape[1]
        elif X.shape[1] != num_features:
            print(f"⚠️ Несовпадение числа признаков в {base}, пропущен")
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

    print(f"\n✅ Готово. Сохранено {total_rows} строк, {len(all_classes)} классов, {num_features} признаков → {output_file}")


# Пример вызова:
build_dataset_file(
    npy_dir="output.lda",
    output_file="lda.csv",
    step=1   # ← можно менять шаг
)
