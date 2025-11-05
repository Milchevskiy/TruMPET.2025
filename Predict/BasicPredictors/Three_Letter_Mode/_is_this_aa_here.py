import math
from typing import List
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants
from _is_this_aa_here_numba import _is_this_aa_here_numba


class _is_this_aa_here:
#    def __init__(self, task_sting):
    def __init__(self, task_string: str, _Common_constants):

        self.task_string = task_string
        words = task_string.split()  # Разделяет по пробелам
        self.aa=words[1]

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


    def calc(self, position_in_chain):
      if self.aa_sequence_three_letter[position_in_chain] == self.aa:
          return 1.0
      else:
          return 0.0


    def calc_vectorized(self) -> np.ndarray:
      seq_array = np.array(self.aa_sequence_three_letter)
      return _is_this_aa_here_numba(seq_array, self.aa)

    def get_task_string(self):
       return self.task_string
