import math
from math import sqrt
from scipy.stats import t

def prob_by_student(t_stat, n):
    """
    Вычисляет двустороннее p-значение для t-статистики,
    используя t-распределение Стьюдента с n степенями свободы.
    
    Параметры:
        t_stat (float): значение t-статистики.
        n (int): число степеней свободы (в данном случае, передаётся casenum).
        
    Возвращает:
        float: двустороннее p-значение.
    """
    if n <= 0:
        return 0
    # Вычисляем двустороннее p-значение по t-распределению
    p_value = t.sf(abs(t_stat), n) * 2
    return p_value
