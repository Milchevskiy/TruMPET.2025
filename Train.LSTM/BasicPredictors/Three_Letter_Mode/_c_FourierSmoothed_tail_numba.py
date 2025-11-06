from numba import njit
import numpy as np

@njit
def _c_FourierSmoothed_tail_numba(seq_len, point_number, left_border, right_border,
                           aaindex_array, unk_value, weights, coef):
    A = np.zeros(seq_len, dtype=np.float32)
    B = np.zeros(seq_len, dtype=np.float32)

    for pos in range(seq_len):
        a_val = 0.0
        b_val = 0.0
        for k in range(len(weights)):
            offset = k + left_border
            ii = pos + offset
            if 0 <= ii < seq_len:
                val = aaindex_array[ii]
            else:
                val = unk_value

            x = ii * coef
            w = weights[k]
            a_val += val * np.sin(x) * w
            b_val += val * np.cos(x) * w

        A[pos] = a_val
        B[pos] = b_val

    return A, B
