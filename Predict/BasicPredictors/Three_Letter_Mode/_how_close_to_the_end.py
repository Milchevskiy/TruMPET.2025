import math
from typing import List
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants

from _is_this_aa_here_numba import _is_this_aa_here_numba

#_how_close_to_the_end N 3

class _how_close_to_the_end:
#    def __init__(self, task_sting):
    def __init__(self, task_string: str, _Common_constants):

        self.task_string = task_string
        words = task_string.split()  # Разделяет по пробелам

        self.N_or_C   = words[1]
        self.zone     = int(words[2])

        if self.N_or_C not in ("N", "C"):
            print(f"❌ Invalid zone specified: '{self.zone}'. Please choose either 'N' (N-end) or 'C' (C-end).")
            sys.exit(1)

        if not (isinstance(self.zone, int) and self.zone >= 0):
           print(f"❌ Invalid zone: {self.zone}. Must be a positive integer.")
           sys.exit(1)

        self.common = _Common_constants
#        self.aaindex_data=self.common.get_aaindex_data()

        self.aa_sequence=''
        self.aa_sequence_three_letter = []

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
#      Обновляет последовательность аминокислот.
      self.aa_sequence = aa_sequence
      self.aa_sequence_three_letter = aa_sequence_three_letter

      # Можем добавить валидацию:
      if len(aa_sequence) != len(aa_sequence_three_letter):
          raise ValueError("Длина однобуквенной и трёхбуквенной последовательностей не совпадает")

      self.aa_length = len(aa_sequence)


    def calc(self, position_in_chain):
       """
       Вычисляет значение на основе позиции в цепочке и ориентации зоны.
       Если позиция в цепочке находится в пределах зоны с конца (для N или C конца),
       возвращает 1.0, иначе — 0.0.
       """
       # Проверяем, если позиция в пределах зоны для N- конца
       if position_in_chain <= self.zone and self.N_or_C == "N":
           return 1.0
       # Проверяем, если позиция в пределах зоны для C- конца
       elif position_in_chain > self.aa_length - self.zone - 1 and self.N_or_C == "C":
           return 1.0
       else:
       # Если позиция не попадает в допустимый диапазон
           return 0.0

    def calc_vectorized(self) -> np.ndarray:
       """
       Векторизированная версия функции для вычисления значений на основе позиции в цепочке и ориентации зоны.
       Возвращает массив значений 1.0 или 0.0 для каждой позиции в цепочке.
       """
       # Массив индексов для проверки
       position_in_chain = np.arange(self.aa_length)

       # Проверка для N-конца
       condition_N = (position_in_chain <= self.zone) & (self.N_or_C == "N")

       # Проверка для C-конца
       condition_C = (position_in_chain > self.aa_length - self.zone - 1) & (self.N_or_C == "C")

       # Комбинируем условия для получения итогового результата
       result = np.where(condition_N | condition_C, 1.0, 0.0)
       return result

    def get_task_string(self):
       return self.task_string

