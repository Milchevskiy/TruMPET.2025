import math
from typing import List
import logging

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants

from DegeneratePredictorGenerator import DegeneratePredictorGenerator

#_DegenerateSequencePredictor _PB_W3_trivial -2 GPAGP


class _DegenerateSequencePredictor:
    def __init__(self, task_string: str, _Common_constants):
        self.task_string = task_string
        self.common = _Common_constants

        words = task_string.split()
        self.frequency_map_name = words[1]
        self.shift      = int(words[2])
        self.keyword            = words[3]

#        self.power = float(words[3]) if len(words) > 3 else 1.0

        self.degenerate_array: List[int] = []

        self.common.add_DSP_item(self.frequency_map_name)

        self.aa_sequence=''
        self.aa_sequence_three_letter = []  

#        path_to_generator = "../FrequencyExtrapolation/" + self.frequency_map_name + "/current.degeneration"
#        self.freq_extrap = DegeneratePredictorGenerator("../FrequencyExtrapolation/_PB_W3_trivial/path_to_generator")

        self.freq_extrap = self.common.DSP_map_dict[self.frequency_map_name]

        window_length    = len(self.freq_extrap.degeneration)  
        if len(self.keyword) != window_length:
            print(f'❌ window_length ({window_length}) does not coincide with word length ({len(self.keyword)})')
            sys.exit(1)  # Завершает программу с кодом ошибки 1

        #print("✅ Покинули Конструктор увы")

        keyword_array = self.freq_extrap.translate_sequence_to_degenerate_array(self.keyword)
        self.keyindex = keyword_array[window_length//2]

        #self.predictor = [1 if x == self.keyindex else 0 for x in degenerate_array]


    def get_keyindex(self):
        return self.keyindex
    def get_degenerate_array(self):
        return self.degenerate_array

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):

        """Обновляет последовательность и строит degenerate_array через объект FrequencyExtrapolation."""
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

#        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        self.freq_extrap.refresh_sequence_and_stuff(aa_sequence)
        self.degenerate_array = self.freq_extrap.translate_sequence_to_degenerate_array(aa_sequence)

#        self.predictor = (self.degenerate_array == self.keyindex).astype(int)
        self.predictor = [1 if x == self.keyindex else 0 for x in self.degenerate_array]


    def calc(self, position_in_chain: int):
         index = position_in_chain + self.shift
         if index < 0  or index >= len (self.aa_sequence):
            return 0
         else:
            return self.predictor [ position_in_chain + self.shift ]

    def calc_vectorized(self) -> np.ndarray:
         seq_len = len(self.aa_sequence)
         shift = self.shift
         predictor = np.array(self.predictor)

         indices = np.arange(seq_len) + shift
         valid_mask = (indices >= 0) & (indices < seq_len)

         values = np.zeros(seq_len, dtype=int)
         values[valid_mask] = predictor[indices[valid_mask]]
         return values

    def get_task_string(self):
       return self.task_string
