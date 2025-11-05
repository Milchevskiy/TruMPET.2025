import math
from typing import List
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np

from gaussian_weights import gaussian_weights
from exponent_factor import exponent_factor

#import logging

# Добавляем путь до модуля FrequencyExtrapolation (если не в sys.path)
#sys.path.append(os.path.join(os.path.dirname(__file__), 'FrequencyExtrapolation'))

from _CommonConstants import _CommonConstants
from _c_FourierSmoothed_tail_numba import _c_FourierSmoothed_tail_numba

#_c_FourierSmoothed_tail          FAUJ880105     1.3    3        1

class _c_FourierSmoothed_tail:
#    def __init__(self, task_string):
    def __init__(self, task_string: str, Common_constants):

        self.task_string = task_string
        self.common = Common_constants
        self.aaindex_data=self.common.get_aaindex_data()

        self.aaindex_data_tri_letter=self.common.get_aaindex_data_tri_letter()

        words = task_string.split()  # Разделяет по пробелам
        self.aaindex_property_name=words[1]
        self.period_lenfth_ = float(words[2])
        self.period_number_ = float(words[3])
        self.power_         = float(words[4])

        self.left_border_  = int (- self.period_number_ * self.period_lenfth_ /2) 
        self.right_border_ = int (  self.period_number_ * self.period_lenfth_ /2) 
        self.point_number_ =   self.right_border_ - self.left_border_
        
        self.aa_sequence = ''
        self.aa_sequence_three_letter = []  

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
#      Обновляет последовательность аминокислот.
      self.aa_sequence = aa_sequence
      self.aa_sequence_three_letter = aa_sequence_three_letter

      # Можем добавить валидацию:
      if len(aa_sequence) != len(aa_sequence_three_letter):
          raise ValueError("Длина однобуквенной и трёхбуквенной последовательностей не совпадает")

        
      
    def calc(self,position_in_chain):
        Coefficient = 2*math.pi / self.period_lenfth_
        pre_start	= position_in_chain + self.left_border_
        pre_end		= position_in_chain + self.right_border_

#        seq_len         = len(self.aa_sequence)
        seq_len = len(self.aa_sequence_three_letter)

        start	= position_in_chain + self.left_border_
        end     = position_in_chain + self.right_border_

        property_dict = self.aaindex_data_tri_letter.get(self.aaindex_property_name, {})
        unk_value = property_dict.get('UNK', None)

        
        A_value=0.0
        B_value=0.0
        current_value=0.0
        for ii in range(start, end):
           x = ii * Coefficient 
           aa_three = self.aa_sequence_three_letter[ii] if 0 <= ii < seq_len else 'UNK'
           current_value = property_dict.get(aa_three, unk_value)
           factor = exponent_factor(self.point_number_, position_in_chain - ii)

           A_value += current_value * math.sin (x) * factor 
           B_value += current_value * math.cos (x) * factor 

        value = math.sqrt(A_value ** 2 + B_value ** 2)
        
        return value ** self.power_



    def calc_vectorized(self) -> np.ndarray:
      seq_len = len(self.aa_sequence_three_letter)
      property_dict = self.aaindex_data_tri_letter.get(self.aaindex_property_name, {})
      unk_value = property_dict.get('UNK', 0.0)

      # Сопоставляем аминокислоты с числовыми значениями
      aa_props = np.zeros(seq_len, dtype=np.float32)
      for i, aa in enumerate(self.aa_sequence_three_letter):
        aa_props[i] = property_dict.get(aa, unk_value)

      window = np.arange(self.left_border_, self.right_border_)
      weights = gaussian_weights(self.point_number_, window)
      coef = 2 * np.pi / self.period_lenfth_

      A, B = _c_FourierSmoothed_tail_numba(
        seq_len=seq_len,
        point_number=self.point_number_,
        left_border=self.left_border_,
        right_border=self.right_border_,
        aaindex_array=aa_props,
        unk_value=unk_value,
        weights=weights,
        coef=coef
      )

      magnitude = np.sqrt(A ** 2 + B ** 2)
      return np.power(magnitude, self.power_).astype(np.float32)


    def get_task_string(self):
       return self.task_string
