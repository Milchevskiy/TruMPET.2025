import sys
import os
import logging

import numpy as np
from _CommonConstants import _CommonConstants
from _create_predictors_set_common import _create_predictors_set_common

# Добавляем путь до модуля
sys.path.append(os.path.join(os.path.dirname(__file__), 'Three_Letter_Mode'))


import numpy as np

def create_predictors_MLP(objects, aa_sequence, aa_sequence_three_letter, common):
    # Обновляем последовательности у всех объектов
    for obj in objects:
        obj.refresh_sequence(aa_sequence, aa_sequence_three_letter)

    # Длина последовательности (число позиций)
    seq_len = len(aa_sequence_three_letter)
    pr_num = len(objects)

    # Заранее создаём матрицу X: строки — позиции в последовательности,
    # столбцы — разные объекты (предикторы)
    X = np.zeros((seq_len, pr_num), dtype=np.float32)

    # Для каждого объекта вызываем векторизованный метод один раз:
    for kk, obj in enumerate(objects):
        # calc_vectorized() внутри класса знает последовательность и должна вернуть np.ndarray длины seq_len
        values = obj.calc_vectorized()

        # Проверка, что вернулся массив нужного размера
        if values.shape[0] != seq_len:
            raise ValueError(
                f"Ожидали ровно {seq_len} значений от calc_vectorized(), "
                f"а получили {values.shape[0]}"
            )

        # Кладём всё сразу в k-ый столбец
        X[:, kk] = values

    return X

