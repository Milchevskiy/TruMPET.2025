import math
import numpy as np
from typing import Union, Tuple

def calc_dispersion_and_average_by_known_sums(
    sum_x: Union[float, np.ndarray],
    sum_x2: Union[float, np.ndarray],
    n: Union[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет среднее и стандартное отклонение по известным суммам и количеству наблюдений.
    
    Формулы:
        average = sum_x / n
        variance = (sum_x2 - sum_x^2 / n) / (n - 1)
    
    Параметры:
        sum_x  — сумма значений
        sum_x2 — сумма квадратов значений
        n      — количество наблюдений (целое число или массив)
    
    Возвращает:
        average, std (вектор или скаляр)
    """
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_x2 = np.asarray(sum_x2, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    # Чтобы избежать деления на 0 и логических ошибок
    valid = n > 1
    average = np.zeros_like(sum_x, dtype=np.float64)
    std = np.zeros_like(sum_x, dtype=np.float64)

    # Вычисляем только для тех, у кого n > 1
    average[valid] = sum_x[valid] / n[valid]
    variance = np.zeros_like(sum_x, dtype=np.float64)
    variance[valid] = (sum_x2[valid] - (sum_x[valid] ** 2) / n[valid]) / (n[valid] - 1)
    
    # Убираем возможные отрицательные значения из-за численных ошибок
    variance = np.maximum(variance, 0.0)
    std[valid] = np.sqrt(variance[valid])

    # Если всё было скалярами — возвращаем скаляры
    if np.isscalar(sum_x) and np.isscalar(sum_x2) and np.isscalar(n):
        return float(average[()]), float(std[()])
    
    return average, std
