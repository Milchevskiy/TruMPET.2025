from numba import njit
import numpy as np

@njit
def _T_statistics_single_numba(
    occur, av1_loc, s1_loc, average_glo, sigma_glo, total, power
):
    """
    Ускоренный расчёт t-статистики и вероятности превышения среднего.
    Используется приближённая сигмоида вместо точного распределения Стьюдента.
    """
    seq_len = occur.shape[0]
    result = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        casenum = occur[i]
        if casenum >= 2:
            av = av1_loc[i]
            s = s1_loc[i]
            avg_loc = av / casenum
            variance_loc = (s - (av * av / casenum)) / (casenum - 1) if casenum > 1 else 0.0
            sigma_loc = np.sqrt(variance_loc) if variance_loc > 0 else 0.0

            pooled_var = sigma_loc ** 2 / casenum
            t_value = 0.0
            prob = 0.0
            if pooled_var > 0:
                t_value = (avg_loc - average_glo) / np.sqrt(pooled_var)
                prob = 1.0 / (1.0 + np.exp(-t_value))  # аппроксимация через сигмоиду
            result[i] = prob ** power
        else:
            result[i] = 0.0

    return result
