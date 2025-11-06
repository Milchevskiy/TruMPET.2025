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

class _T_statistics_single_max:
    def __init__(self, task_string: str, common_constants):
        self.task_string = task_string
        self.common = common_constants

        words = task_string.split()
        self.frequency_map_name = words[1]
        self.claster_index = int(words[2])
        self.max_min_mode = words[3]
        self.power = float(words[4]) if len(words) > 4 else 1.0

        self.aa_sequence=''
        self.aa_sequence_three_letter = []

        self.degenerate_array: List[int] = []

        self.common.add_frequency_item(self.frequency_map_name)

        # Получаем длину фрагмента из объекта frequency map
        self.window_length = self.common.frequency_map_dict.get(self.frequency_map_name).get_fragment_length()

        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):
        """
        Обновляет последовательность и строит degenerate_array через объект FrequencyExtrapolation.
        """
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

#        freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
#        freq_extrap.refresh_sequence_and_stuff(aa_sequence)

#        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        self.freq_extrap.refresh_sequence_and_stuff(aa_sequence)

        self.degenerate_array = self.freq_extrap.translate_sequence_to_degenerate_array(aa_sequence)
        self.is_single_ready_value_setted=False
        self.single_ready_value=0.0


    def calc(self, position_in_chain: int):
        """
        Вычисляет значение статистики для заданной позиции в цепочке.
        """
        if self.is_single_ready_value_setted==True:
           return self.single_ready_value

        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        #occurence = freq_extrap.respect_occurrence
        occurence = self.freq_extrap.get_respect_occurence()

        total_sample_size = self.freq_extrap.get_total_sample_size()


        seq_len = len(self.aa_sequence)
#        pull_obj = self.common.frequency_map_dict[self.frequency_map_name]

        #t_dist=freq_extrap.get_tot_squared_distance_to_clusters_sum()

       # global_array_av1 = freq_extrap.get_tot_distance_to_clusters_sum()
       # global_array_s1 = freq_extrap.get_tot_squared_distance_to_clusters_sum()

       # qqq=global_array_av1[self.claster_index]

        # Глобальные суммы
        av1_glo = self.freq_extrap.get_tot_distance_to_clusters_sum()[self.claster_index]
        s1_glo = self.freq_extrap.get_tot_squared_distance_to_clusters_sum()[self.claster_index]
        casenum_glo = self.freq_extrap.get_total_sample_size()

        # Вычисляем глобальное среднее и sigma
        average_glo, sigma_glo = calc_dispersion_and_average_by_known_sums(av1_glo, s1_glo, casenum_glo)

        value_array = []
        # Перебираем все возможные фрагменты длиной window_length
        for kk in range(seq_len - self.window_length + 1):
            av1_loc = self.freq_extrap.distance_to_clusters_sum[kk][self.claster_index]
            s1_loc = self.freq_extrap.squared_distance_to_clusters_sum[kk][self.claster_index]
            casenum_loc = self.freq_extrap.get_respect_occurence()[kk]

            # Вычисляем локальное среднее и sigma
            if casenum_loc > 0:
                average_loc, sigma_loc = calc_dispersion_and_average_by_known_sums(av1_loc, s1_loc, casenum_loc)
            else:
                average_loc, sigma_loc = average_glo, sigma_glo

            # Чтобы избежать деления на 0
            t_val = (average_glo - average_loc) * math.sqrt(casenum_loc) / sigma_loc if sigma_loc != 0 else 0

            if casenum_loc == 1 or t_val == 0.0:
                value_array.append(0.0)
            else:
                #current_prob = prob_by_student(t_val, casenum_loc)
                current_prob = probability_known_mean_greater_mean2(average_glo, average_loc, sigma_loc, casenum_loc)
                value_array.append(current_prob)
        value_array.sort()
        mode = self.max_min_mode.lower()
        if mode == "min":
            result = value_array[0]
        elif mode == "max":
            result = value_array[-1]
        elif mode == "average":
            avg, _ = calc_dispersion_and_average(value_array)
            result = avg
        elif mode == "dispersion":
            _, dispersion = calc_dispersion_and_average(value_array)
            result = dispersion
        else:
            result = 0

        self.is_single_ready_value_setted=True
        self.single_ready_value = math.pow(result, self.power)

        return self.single_ready_value

    def calc_vectorized(self) -> np.ndarray:
      if not self.is_single_ready_value_setted:
          # Тригерим расчёт единственного значения
          _ = self.calc(0)

    # Возвращаем вектор из одинаковых значений
      return np.full(len(self.aa_sequence), self.single_ready_value, dtype=np.float32)


    def get_task_string(self):
       return self.task_string
