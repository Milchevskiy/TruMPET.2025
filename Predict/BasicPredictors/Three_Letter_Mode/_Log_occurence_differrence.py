import math
from typing import List
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from _CommonConstants import _CommonConstants
from _Log_occurence_differrence_numba import _Log_occurence_differrence_numba

#_Log_occurence_differrence  W5_01234_dx2 0     1
class _Log_occurence_differrence:
    def __init__(self, task_string: str, _Common_constants):
        self.task_string = task_string
        self.common = _Common_constants

        words = task_string.split()
        self.frequency_map_name = words[1]
        self.claster_index = int(words[2])
        self.power = float(words[3]) if len(words) > 3 else 1.0

        self.degenerate_array: List[int] = []

        self.common.add_frequency_item(self.frequency_map_name)

        self.aa_sequence=''
        self.aa_sequence_three_letter = []

        self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

    def refresh_sequence(self, aa_sequence: str, aa_sequence_three_letter: List[str]):

        """Обновляет последовательность и строит degenerate_array через объект FrequencyExtrapolation."""
        self.aa_sequence = aa_sequence
        self.aa_sequence_three_letter = aa_sequence_three_letter

#        freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]
        self.freq_extrap.refresh_sequence_and_stuff(aa_sequence)
        self.degenerate_array = self.freq_extrap.translate_sequence_to_degenerate_array(aa_sequence)

    def calc(self, position_in_chain: int) -> float:
        """Python-эквивалент C++ метода calc_value с логами и безопасностью."""
        self.freq_extrap = self.common.frequency_map_dict.get(self.frequency_map_name)

        if not self.freq_extrap:
            logging.error(f"Карта '{self.frequency_map_name}' не найдена в frequency_map_dict.")
            return 0.0

        #occurence = freq_extrap.respect_occurrence
        occurence = self.freq_extrap.get_respect_occurence()

        total_sample_size = self.freq_extrap.get_total_sample_size()

        if position_in_chain < 0 or position_in_chain >= len(occurence):
            logging.warning(f"Позиция {position_in_chain} выходит за границы последовательности.")
            return 0.0

        ocu = occurence[position_in_chain]
        current_array = self.freq_extrap.distance_to_clusters_sum[position_in_chain]
        global_array = self.freq_extrap.tot_distance_to_clusters_sum

        if self.claster_index >= len(current_array) or self.claster_index >= len(global_array):
            logging.warning(f"Кластерный индекс {self.claster_index} выходит за границы.")
            return 0.0

        distance_to_cluster_sum = current_array[self.claster_index]
        tot_distance_to_clusters_sum = global_array[self.claster_index]

        logging.debug(f"[calc_value] Position {position_in_chain}")
        logging.debug(f"  Occurrence: {ocu}")
        logging.debug(f"  Distance (local): {distance_to_cluster_sum}")
        logging.debug(f"  Distance (global): {tot_distance_to_clusters_sum}")
        logging.debug(f"  Total sample size: {total_sample_size}")

        if ocu > 0 and total_sample_size > 0:
            log_term = math.log(1 + math.log(1 + ocu))
            delta = (tot_distance_to_clusters_sum / total_sample_size) - (distance_to_cluster_sum / ocu)
            sum_value = log_term * delta
        else:
            sum_value = 0.0

        result = math.pow(sum_value, self.power)

        logging.debug(f"  Log term: {log_term if ocu > 0 else 'N/A'}")
        logging.debug(f"  Delta: {delta if ocu > 0 else 'N/A'}")
        logging.debug(f"  Result: {result}")

        return result

    def calc_vectorized(self) -> np.ndarray:
      self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

      occurence = np.array(self.freq_extrap.get_respect_occurence(), dtype=np.float32)
      local = np.array(self.freq_extrap.distance_to_clusters_sum, dtype=np.float32)[:, self.claster_index]
      global_ = float(self.freq_extrap.tot_distance_to_clusters_sum[self.claster_index])
      total_sample_size = float(self.freq_extrap.get_total_sample_size())

      with np.errstate(divide='ignore', invalid='ignore'):
          log_term = np.log1p(np.log1p(occurence))
          delta = (global_ / total_sample_size) - (local / occurence)
          result = np.power(log_term * delta, self.power)
          result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

      return result.astype(np.float32)


    def calc_vectorized(self) -> np.ndarray:
      self.freq_extrap = self.common.frequency_map_dict[self.frequency_map_name]

      occurence = np.array(self.freq_extrap.get_respect_occurence(), dtype=np.float32)
      local = np.array(self.freq_extrap.distance_to_clusters_sum, dtype=np.float32)[:, self.claster_index]
      global_ = float(self.freq_extrap.tot_distance_to_clusters_sum[self.claster_index])
      total_sample_size = float(self.freq_extrap.get_total_sample_size())

      return _Log_occurence_differrence_numba(
        occurence=occurence,
        local=local,
        global_=global_,
        total_sample_size=total_sample_size,
        power=self.power
      )


    def get_task_string(self):
       return self.task_string


