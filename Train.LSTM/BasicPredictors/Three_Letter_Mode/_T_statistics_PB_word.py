#!/usr/bin/env python3
import math
from math import sqrt
from scipy.stats import t
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

from _T_statistics_single_window_numba import _T_statistics_single_window_numba

from PB_word_to_indexes import PB_word_to_indexes

# _T_statistics_PB_word  DDDBM 0 2'

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


#  _T_statistics_PB_word   PB_W5_noDEG_ZIP   DDBMM  2


class _T_statistics_PB_word:
    def __init__(self, task_string: str, common_constants):
        self.task_string = task_string
        self.common = common_constants

        words = task_string.split()
        self.frequency_map_name = words[1]
        self.PB_word            = words[2]
        self.Shift              = int(words[3])
        self.power = float(words[4]) if len(words) > 3 else 1.0


        self.length_PB_word = len(self.PB_word)
        self.left_border  = - int(self.length_PB_word/2) + self.Shift
        self.right_border =   int(self.length_PB_word/2) + self.Shift + 1

        #self.max_min_mode = words[3]

        self.claster_index_set =  PB_word_to_indexes(self.PB_word)

        self.aa_sequence=''
        self.aa_sequence_three_letter = []

        self.degenerate_array: List[int] = []

        self.common.add_frequency_item(self.frequency_map_name)

        # Используем common для получения нужного индекса
#        if self.frequency_map_name in self.common.frequency_name_to_pull_index_:
#            self.index_in_frequency_pull = self.common.frequency_name_to_pull_index_[self.frequency_map_name]
#        else:
#            print(f"{self.frequency_map_name} _T_statistics_PB_word() ERROR: can't associate Frequency_extrapolation object for name")
#            exit(-1)

        # Получаем длину фрагмента из объекта frequency map
        self.window_length = self.common.frequency_map_dict.get(self.frequency_map_name).get_fragment_length()


        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):

        """Обновляет последовательность и строит degenerate_array через объект FrequencyExtrapolation."""
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

        #self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        self.freq_extrap.refresh_sequence_and_stuff(aa_sequence)
        self.degenerate_array = self.freq_extrap.translate_sequence_to_degenerate_array(aa_sequence)




    def calc(self, position_in_chain: int):
       """
       Вычисляет значение статистики для заданной позиции в цепочке
       с учетом окна [position_in_chain + left_border, position_in_chain + right_border).
       """
       occurence = self.freq_extrap.get_respect_occurence()
       total_sample_size = self.freq_extrap.get_total_sample_size()
       seq_len = len(self.aa_sequence)

       pre_start = position_in_chain + self.left_border
       pre_end   = position_in_chain + self.right_border

       start = max(0, pre_start)
       end   = min(pre_end, seq_len)

# было в _T_statistics_single_window
#       cl = self.claster_index
#       av1_glo = self.freq_extrap.get_tot_distance_to_clusters_sum()[cl]
#       s1_glo = self.freq_extrap.get_tot_squared_distance_to_clusters_sum()[cl]
#       average_glo, sigma_glo = calc_dispersion_and_average_by_known_sums(av1_glo, s1_glo, total_sample_size)


       # Берём один раз «полные» массивы сумм по всем кластерам
       tot_distances = self.freq_extrap.get_tot_distance_to_clusters_sum()
       tot_squared   = self.freq_extrap.get_tot_squared_distance_to_clusters_sum()

       average_glo = []
       sigma_glo  = []
       # Вариант 1: через обычный цикл
       for cl in self.claster_index_set:
         average_, sigma_ = calc_dispersion_and_average_by_known_sums(tot_distances[cl], tot_squared[cl], total_sample_size)
         average_glo.append(average_)
         sigma_glo.append(sigma_)
       # Конвертируем в numpy-массивы (если вам действительно нужен ndarray)
       average_glo_arr = np.array(average_glo, dtype=np.float32)
       s1_glo_arr      = np.array(sigma_glo,   dtype=np.float32)

       sum_of_probabilities_within_a_window = 0.0

       kk=0
       for ii in range(start, end):
          cl=self.claster_index_set[kk]
          av1_loc = self.freq_extrap.distance_to_clusters_sum[ii][cl]
          s1_loc = self.freq_extrap.squared_distance_to_clusters_sum[ii][cl]
          casenum_loc = occurence[ii]

          if casenum_loc < 2:
            current_prob = 0.0
          else:
            average_loc, sigma_loc = calc_dispersion_and_average_by_known_sums(av1_loc, s1_loc, casenum_loc)
            current_prob = probability_known_mean_greater_mean2(average_glo_arr[kk], average_loc, sigma_loc, casenum_loc)
            kk = kk + 1

          sum_of_probabilities_within_a_window += ( 1 - current_prob )

       return math.pow(sum_of_probabilities_within_a_window, self.power)


    def calc_vectorized(self) -> np.ndarray:
        """
        Для каждой позиции i в цепочке считает
        sum_{k=0..w-1} (1 - P(μ_global[k] > μ_local[i+left+k]))
        и в конце поднимает в степень self.power.
        """
        freq = self.freq_extrap

        # Векторы из FrequencyExtrapolation
        occur_arr = np.array(freq.get_respect_occurence(), dtype=np.int32)            # (seq_len,)
        dist_mat  = np.array(freq.distance_to_clusters_sum, dtype=np.float32)         # (seq_len, n_clusters)
        sq_mat    = np.array(freq.squared_distance_to_clusters_sum, dtype=np.float32) # (seq_len, n_clusters)
        tot_dist  = np.array(freq.get_tot_distance_to_clusters_sum(), dtype=np.float32)      # (n_clusters,)
        tot_sq    = np.array(freq.get_tot_squared_distance_to_clusters_sum(), dtype=np.float32)  # (n_clusters,)
        total_sz  = freq.get_total_sample_size()

        cl_idx = np.array(self.claster_index_set, dtype=np.int32)  # (w,)
        left, right, power = self.left_border, self.right_border, self.power
        seq_len = occur_arr.shape[0]
        w = cl_idx.size

        # Глобальные μ и σ для каждого кластера в наборе
        μ_glo = tot_dist[cl_idx] / total_sz
        var_glo = (tot_sq[cl_idx] - tot_dist[cl_idx]**2 / total_sz) / (total_sz - 1)
        σ_glo = np.sqrt(var_glo)

        # Результирующий вектор (сумма по окну для каждой позиции)
        result = np.zeros(seq_len, dtype=np.float32)

        # Цикл по смещениям внутри окна (обычно окно — длина PB_word, небольшая)
        for k, cl in enumerate(cl_idx):
            offset = left + k
            positions = np.arange(seq_len) + offset

            # валидные позиции в цепочке
            mask_valid = (positions >= 0) & (positions < seq_len)
            idx_valid  = positions[mask_valid]

            occ = occur_arr[idx_valid]
            # для тех позиций, где случайных наблюдений < 2, вероятность = 0
            mask2 = occ >= 2

            # локальные суммы и квадраты
            loc_dist = dist_mat[idx_valid, cl]
            loc_sq   = sq_mat[idx_valid,   cl]

            # локальные μ и σ (на тех, где occ>=2)
            μ_loc = np.zeros_like(occ, dtype=np.float32)
            σ_loc = np.zeros_like(occ, dtype=np.float32)
            μ_loc[mask2] = loc_dist[mask2] / occ[mask2]
            var_loc = np.zeros_like(occ, dtype=np.float32)
            var_loc[mask2] = (loc_sq[mask2] - loc_dist[mask2]**2 / occ[mask2]) / (occ[mask2] - 1)

            neg_mask = mask2 & (var_loc < 0.0)
            if np.any(neg_mask):
              # print(var_loc[neg_mask])
              var_loc[neg_mask] = 0.0
            σ_loc[mask2] = np.sqrt(var_loc[mask2])

            # p = P(μ_glo > μ_loc) векторно
            probs = np.zeros_like(occ, dtype=np.float32)
            probs[mask2] = probability_known_mean_greater_mean2_vectorized(
                μ_glo[k], μ_loc[mask2], σ_loc[mask2], occ[mask2]
            )

            # суммируем (1 - p)
            result[mask_valid] += (1.0 - probs)

        # финальный степенной рост
        return np.power(result, power)

    def get_freq_extrap(self):
        return self.freq_extrap

    def get_task_string(self):
       return self.task_string

    def letters_to_indices(self, letters):
        result = []
        for ch in letters:
            if ch not in self.PB_to_index:
                # Сообщаем о проблеме и завершаем программу
                print(f"Неизвестный символ: {ch!r}. Программа прерывается.")
                sys.exit(1)
            result.append(self.PB_to_index[ch])
        return result
