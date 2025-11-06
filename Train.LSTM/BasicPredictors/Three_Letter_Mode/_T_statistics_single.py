import math
from math import sqrt
from scipy.stats import t  # для расчёта p-значения по распределению Стьюдента
from typing import List
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from calc_dispersion_and_average_by_known_sums import calc_dispersion_and_average_by_known_sums
from prob_by_student import prob_by_student 
from _CommonConstants import _CommonConstants 

from probability_known_mean_greater_mean2 import probability_known_mean_greater_mean2
from probability_known_mean_greater_mean2_vectorized import probability_known_mean_greater_mean2_vectorized

from _T_statistics_single_numba import _T_statistics_single_numba

#'_T_statistics_single PB_W7_tail_GP 0 1'

# Функция для расчёта среднего и дисперсии по списку значений (если требуется)
def calc_dispersion_and_average(values: List[float]):
    n = len(values)
    if n == 0:
        return 0, 0
    avg = sum(values) / n
    if n > 1:
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
    else:
        variance = 0
    return avg, variance

class _T_statistics_single:
    def __init__(self, task_string: str, _CommonConstants):
        self.task_string = task_string
        self.common = _CommonConstants

        words = task_string.split()
        self.frequency_map_name = words[1]
        self.claster_index = int(words[2])
        #self.max_min_mode = words[3]
        self.power = float(words[3]) if len(words) > 3 else 1.0

        self.aa_sequence=''
        self.aa_sequence_three_letter = []  

        self.degenerate_array: List[int] = []

        self.common.add_frequency_item(self.frequency_map_name)
        
        # Получаем длину фрагмента из объекта frequency map
        self.window_length = self.common.frequency_map_dict.get(self.frequency_map_name).get_fragment_length()

        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):

        """Обновляет последовательность и строит degenerate_array через объект FrequencyExtrapolation."""
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

#        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        self.freq_extrap.refresh_sequence_and_stuff(aa_sequence)
        self.degenerate_array = self.freq_extrap.translate_sequence_to_degenerate_array(aa_sequence)
        




    def calc(self, position_in_chain: int):
      occurence = self.freq_extrap.get_respect_occurence()
      total_sample_size = self.freq_extrap.get_total_sample_size()

      cl = self.claster_index
      av1_glo = self.freq_extrap.get_tot_distance_to_clusters_sum()[cl]
      s1_glo = self.freq_extrap.get_tot_squared_distance_to_clusters_sum()[cl]
      average_glo, sigma_glo = calc_dispersion_and_average_by_known_sums(av1_glo, s1_glo, total_sample_size)

      av1_loc = self.freq_extrap.distance_to_clusters_sum[position_in_chain][cl]
      s1_loc = self.freq_extrap.squared_distance_to_clusters_sum[position_in_chain][cl]
      casenum_loc = occurence[position_in_chain]

      if casenum_loc < 2:
          current_prob = 0.0
      else:
          average_loc, sigma_loc = calc_dispersion_and_average_by_known_sums(av1_loc, s1_loc, casenum_loc)
          current_prob = probability_known_mean_greater_mean2(average_glo, average_loc, sigma_loc, casenum_loc)

      return math.pow(current_prob, self.power)

    def calc_vectorized(self) -> np.ndarray:
      cl = self.claster_index
      power = self.power

      total = self.freq_extrap.get_total_sample_size()
      av1_glo = self.freq_extrap.get_tot_distance_to_clusters_sum()[cl]
      s1_glo = self.freq_extrap.get_tot_squared_distance_to_clusters_sum()[cl]
      average_glo, sigma_glo = calc_dispersion_and_average_by_known_sums(av1_glo, s1_glo, total)

      occur = np.array(self.freq_extrap.get_respect_occurence(), dtype=np.int32)
      av1_loc = np.array([row[cl] for row in self.freq_extrap.distance_to_clusters_sum], dtype=np.float32)
      s1_loc = np.array([row[cl] for row in self.freq_extrap.squared_distance_to_clusters_sum], dtype=np.float32)

      result = np.zeros(len(occur), dtype=np.float32)
      mask = occur >= 2

      av_masked = av1_loc[mask]
      s_masked = s1_loc[mask]
      casenum_masked = occur[mask]

      avg_loc, sigma_loc = calc_dispersion_and_average_by_known_sums(av_masked, s_masked, casenum_masked)
      probs = probability_known_mean_greater_mean2_vectorized(
          average_glo, avg_loc, sigma_loc, casenum_masked
      )

      result[mask] = np.power(probs, power)
      return result

    def get_task_string(self):
       return self.task_string
