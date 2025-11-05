from numba import njit
import numpy as np

@njit
def _Log_occurence_differrence_numba_INV(occurence, local, global_, total_sample_size, power):
    """
    Ускоренная реализация метода calc_vectorized() для _Log_occurence_differrence.
    Все массивы должны быть NumPy-массивами float32.

    Параметры:
    - occurence: вектор встречаемости по позициям
    - local: локальные суммы расстояний до кластера
    - global_: глобальная сумма расстояний
    - total_sample_size: общее количество наблюдений
    - power: степень, в которую возводится результат

    Возвращает:
    - NumPy-массив результата shape (N,)
    """
    seq_len = occurence.shape[0]
    result = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        ocu = occurence[i]
        loc = local[i]

        if ocu > 0 and total_sample_size > 0:
            log_term = np.log(1.0 + np.log(1.0 + ocu))
            delta = (global_ / total_sample_size) - (loc / ocu)
            val = log_term * delta
            result[i] = val ** power
        else:
            result[i] = 0.0

    return result
