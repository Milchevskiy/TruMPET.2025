import math
from scipy.stats import t
import numpy as np

def probability_known_mean_greater_mean2(mu1, mu2, sigma2, n2):
#    Возвращает вероятность того, что mu1 > mu2, 
#    при известном mu1 и выборке (mu2, sigma2, n2) для второй группы.
#
#    Используется t-распределение:
#        t = (mu1 - mu2) / (sigma2 / sqrt(n2))
#        df = n2 - 1


    if n2 < 2 or sigma2 <= 0:
        return 0.0

    stderr = sigma2 / math.sqrt(n2)
    t_stat = (mu1 - mu2) / stderr
    df = n2 - 1

    p_value = 1.0 - t.cdf(t_stat, df)
    return float(p_value)
