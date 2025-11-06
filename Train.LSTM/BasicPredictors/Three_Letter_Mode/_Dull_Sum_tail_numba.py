from numba import njit
import numpy as np

@njit
def _Dull_Sum_tail_numba(seq_len, left_border, right_border, power, aaindex_array, unk_value, weights):
    """
    Быстрая векторизованная реализация функции calc_vectorized для класса _Dull_Sum_tail.

    Параметры:
    - seq_len: длина аминокислотной последовательности
    - left_border, right_border: границы окна
    - power: степень возведения суммы
    - aaindex_array: значения свойства аминокислот (в том же порядке, что и последовательность)
    - unk_value: значение по умолчанию для неизвестных остатков
    - weights: веса (из gaussian_weights)

    Возвращает:
    - Массив значений по всей последовательности.
    """
    values = np.zeros(seq_len, dtype=np.float32)

    for pos in range(seq_len):
        s = 0.0
        for k in range(len(weights)):
            offset = k + left_border
            ii = pos + offset
            if 0 <= ii < seq_len:
                val = aaindex_array[ii]
            else:
                val = unk_value
            s += val * weights[k]
        values[pos] = s ** power

    return values
