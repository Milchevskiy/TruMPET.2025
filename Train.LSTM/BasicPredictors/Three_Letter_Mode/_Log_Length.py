import math
from math import sqrt
from scipy.stats import t
from typing import List
import logging
import sys
import os
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants
class _Log_Length:
    def __init__(self, task_string: str, common_constants):
        self.task_string = task_string
        self.common = common_constants

        # разбор аргументов: если есть второй элемент — используем его как степень
        words = task_string.split()
        self.power = float(words[1]) if len(words) > 1 else 1.0

        self.aa_sequence = ''
        self.aa_sequence_three_letter = []

        self.is_single_ready_value_setted = False
        self.single_ready_value = None

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter
        # сбрасываем, чтобы при смене последовательности значение пересчиталось
        self.is_single_ready_value_setted = False
        self.single_ready_value = None

    def calc(self, position_in_chain: int) -> float:
        """
        Берём логарифм от длины цепи и возводим его в self.power.
        """
        seq_len = len(self.aa_sequence)
        log_len = math.log(seq_len)               # натуральный логарифм
        return math.pow(log_len, self.power)      # (log_len)^power

    def calc_vectorized(self) -> np.ndarray:
        if not self.is_single_ready_value_setted:
            # при первом вызове сохраняем одно значение
            self.single_ready_value = self.calc(0)
            self.is_single_ready_value_setted = True

        # возвращаем массив из одинаковых значений
        return np.full(len(self.aa_sequence),
                       self.single_ready_value,
                       dtype=np.float32)

    def get_task_string(self):
        return self.task_string

