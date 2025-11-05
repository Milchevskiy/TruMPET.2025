import math
from typing import List
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants
from iupred2a_lib import iupred


#from DegeneratePredictorGenerator import DegeneratePredictorGenerator
# _Iupred2 long 1
#  Available types: "long", "short", "glob"


class _Iupred2:
    def __init__(self, task_string: str, _Common_constants):
        self.task_string = task_string
        self.common = _Common_constants

        words = task_string.split()
        self.run_mode = words[1]
        self.power_  = float(words[2])

        self.aa_sequence=''
        self.aa_sequence_three_letter = []

        if self.run_mode not in ['short', 'long', 'glob']:
          sys.exit('Wrong iupred2 option {self.run_mode}')


    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

        result = iupred(self.aa_sequence, self.run_mode)

        if isinstance(result, tuple):
           self.predictor = result[0]  # только список чисел
        else:
           self.predictor = result

        self.predictor = np.array(self.predictor, dtype=np.float32)
        self.predictor = self.predictor ** self.power_



    def calc(self, position_in_chain: int):
         index = position_in_chain# + self.shift
         if index < 0  or index >= len (self.aa_sequence):
            return 0
         else:
            return self.predictor [ position_in_chain ]

    def calc_vectorized(self) -> np.ndarray:
         seq_len = len(self.aa_sequence)
         predictor = np.array(self.predictor, dtype=np.float32)

         values = np.zeros(seq_len, dtype=np.float32)
         values[:] = predictor[:seq_len]  # поскольку индексы от 0 до seq_len-1
         return values

    def get_task_string(self):
       return self.task_string
