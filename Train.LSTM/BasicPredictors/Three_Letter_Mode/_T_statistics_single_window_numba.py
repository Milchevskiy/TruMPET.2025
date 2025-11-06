from numba import njit
import numpy as np

@njit
def _T_statistics_single_window_numba(probs, left, right, power):
    """
    Numba-ускоренная оконная агрегация.

    Для каждой позиции считает сумму значений probs
    в окне [pos + left, pos + right), затем возводит сумму в степень power.

    Parameters:
    - probs: np.ndarray, shape (N,)
    - left: int, смещение начала окна
    - right: int, смещение конца окна
    - power: float, степень

    Returns:
    - np.ndarray, shape (N,)
    """
    seq_len = probs.shape[0]
    result = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        start = i + left
        end = i + right
        s = 0.0
        for j in range(start, end):
            if 0 <= j < seq_len:
                s += probs[j]
        result[i] = s ** power

    return result
