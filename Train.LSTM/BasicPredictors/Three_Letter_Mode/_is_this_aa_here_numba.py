from numba import njit
import numpy as np

@njit
def _is_this_aa_here_numba(seq_array, target_aa):
    """
    Возвращает массив, где 1.0 — если аминокислота в позиции совпадает с target_aa, иначе 0.0

    Параметры:
    - seq_array: массив строк с трёхбуквенными кодами
    - target_aa: строка, искомая аминокислота

    Возвращает:
    - NumPy-массив float32 shape (N,)
    """
    seq_len = len(seq_array)
    result = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        if seq_array[i] == target_aa:
            result[i] = 1.0
    return result
