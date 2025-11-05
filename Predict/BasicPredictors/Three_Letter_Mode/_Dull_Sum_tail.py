from exponent_factor import exponent_factor

from gaussian_weights import gaussian_weights

import math
import numpy as np

from typing import List

from _Dull_Sum_tail_numba import _Dull_Sum_tail_numba

#Dull_Sum_tail WARP780101 -5 6 1

class _Dull_Sum_tail:
#    def __init__(self, task_sting):
    def __init__(self, task_string: str, Common_constants):

        self.task_sting = task_string
        words = task_string.split()  # Разделяет по пробелам
        self.aaindex_property_name=words[1]
        self.left_border_  = int(words[2])
        self.right_border_ = int(words[3])
        self.power_        = float(words[4])
        self.point_number_ =   self.right_border_ - self.left_border_

        self.common = Common_constants
        self.aaindex_data=self.common.get_aaindex_data()

        self.aaindex_data_tri_letter=self.common.get_aaindex_data_tri_letter()


        self.aa_sequence=''
        self.aa_sequence_three_letter = []


# Далее возможна обработка, например, построение эмбеддингов

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
#      Обновляет последовательность аминокислот.
      self.aa_sequence = aa_sequence
      self.aa_sequence_three_letter = aa_sequence_three_letter

      # Можем добавить валидацию:
      if len(aa_sequence) != len(aa_sequence_three_letter):
          raise ValueError("Длина однобуквенной и трёхбуквенной последовательностей не совпадает")


    def calc(self, position_in_chain: int) -> float:
          seq_len = len(self.aa_sequence_three_letter)
          start = position_in_chain + self.left_border_
          end = position_in_chain + self.right_border_

          sum_values = 0.0

          property_dict = self.aaindex_data_tri_letter.get(self.aaindex_property_name, {})
          unk_value = property_dict.get('UNK', None)

          for ii in range(start, end):
             aa_three = self.aa_sequence_three_letter[ii] if 0 <= ii < seq_len else 'UNK'
             current_value = property_dict.get(aa_three, unk_value)
             factor = exponent_factor(self.point_number_, position_in_chain - ii)
             sum_values += current_value * factor

          return sum_values ** self.power_

    def calc_vectorized(self) -> np.ndarray:
      seq_len = len(self.aa_sequence_three_letter)
      property_dict = self.aaindex_data_tri_letter.get(self.aaindex_property_name, {})
      unk_value = property_dict.get('UNK', 0.0)

      # Преобразуем последовательность в массив значений
      aa_props = np.zeros(seq_len, dtype=np.float32)
      for i, aa in enumerate(self.aa_sequence_three_letter):
        aa_props[i] = property_dict.get(aa, unk_value)

      # Генерируем окно и веса
      window = np.arange(self.left_border_, self.right_border_)
      weights = gaussian_weights(self.point_number_, window)

      return _Dull_Sum_tail_numba(
        seq_len=seq_len,
        left_border=self.left_border_,
        right_border=self.right_border_,
        power=self.power_,
        aaindex_array=aa_props,
        unk_value=unk_value,
        weights=weights
      )

    def get_task_string(self):
       return self.task_sting

