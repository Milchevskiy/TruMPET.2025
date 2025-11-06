import numpy as np
from scipy.stats import t

def probability_known_mean_greater_mean2_vectorized(mu1, mu2_array, sigma2_array, n2_array):
    """
    Вычисляет p-value: вероятность того, что mu1 > mu2,
    при известном mu1 и неизвестном mu2 с выборкой (mu2_array, sigma2_array, n2_array).
    
    Формула использует t-распределение:
        t = (mu1 - mu2) / (sigma2 / sqrt(n2))
    
    Возвращает массив p-value shape = mu2_array.shape
    """
    mu2_array = np.asarray(mu2_array)
    sigma2_array = np.asarray(sigma2_array)
    n2_array = np.asarray(n2_array)

    # чтобы избежать деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        stderr = sigma2_array / np.sqrt(n2_array)
        t_stat = (mu1 - mu2_array) / stderr
        df = n2_array - 1

        # t.cdf возвращает P(T <= t), нам нужно P(T > t)
        p_values = 1.0 - t.cdf(t_stat, df)
        p_values = np.clip(p_values, 0.0, 1.0)
        p_values = np.nan_to_num(p_values, nan=0.0, posinf=0.0, neginf=1.0)

    return p_values.astype(np.float32)
