import logging
from pathlib import Path
from typing import List

import numpy as np

import sys
import os

# Добавляем путь до модуля 
sys.path.append(os.path.join(os.path.dirname(__file__), '../FrequencyExtrapolation'))
 

from AminoAcidConverter import AminoAcidConverter
from CurlyCalculusSystem import CurlyCalculusSystem
from aa_sequence_to_index_array import aa_sequence_to_index_array


class DegeneratePredictorGenerator:
    def __init__(self,DSP_name,path_to_frequency_store: str):
        self.position_shift: List[int] = []
        self.degeneration: List[List[str]] = []
        self.index_of_degeneration: np.ndarray
        self.curly_calculus_system: CurlyCalculusSystem
        self.window_size: int
        self.left_border_value: int
        self.right_border_value: int

        degeneration_file = path_to_frequency_store + DSP_name + "/current.degeneration"

        self.read_degeneration(degeneration_file)
        self.assign_window_size_and_borders()
        self.translate_degeneration_to_index()
        self.init_curly_calculus_system()
        self.number_of_elements = self.curly_calculus_system.get_number_of_elements()

    def read_degeneration(self, degeneration_file: str):
        path = Path(degeneration_file)
        if not path.exists():
            raise FileNotFoundError(f"Degeneration file not found: {degeneration_file}")

        with path.open('r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(('/', '#')):
                    continue
                parts = line.split()
                self.position_shift.append(int(parts[0]))
                self.degeneration.append(parts[1:])

    def assign_window_size_and_borders(self):
        self.window_size = len(self.position_shift)
        self.left_border_value = self.position_shift[0]
        self.right_border_value = self.position_shift[-1]

    def translate_degeneration_to_index(self):
        map_size = AminoAcidConverter.get_size_aminoacid_set()
        degeneration_size = len(self.degeneration)
        self.index_of_degeneration = np.full((degeneration_size, map_size), -1, dtype=int)

        for i, group in enumerate(self.degeneration):
            for j, word in enumerate(group):
                for aa in word:
                    aa_index = AminoAcidConverter.aminoacid_to_index(aa)
                    self.index_of_degeneration[i, aa_index] = j

    def init_curly_calculus_system(self):
        bases = [len(group) for group in self.degeneration]
        self.curly_calculus_system = CurlyCalculusSystem(bases)

#    def generate_predictors(self, sequence: str) -> List[int]:
    def translate_sequence_to_degenerate_array(self, sequence: str) -> List[int]:
        crude_index_array = aa_sequence_to_index_array(sequence)
        return self.translate_crude_index_array_to_degenerate_array(crude_index_array)

    def translate_crude_index_array_to_degenerate_array(self, crude_index_array: List[int]) -> List[int]:
        sequence_size = len(crude_index_array)
        degenerate_array = [0] * sequence_size
        index_subset_array = [0] * self.window_size

        for ii in range(sequence_size):
            self.fill_up_appropriate_crude_index_subset(
                self.index_of_degeneration,
                ii,
                self.left_border_value,
                self.right_border_value,
                crude_index_array,
                AminoAcidConverter.get_virtual_residue_index(),
                index_subset_array
            )

            cursor = self.curly_calculus_system.get_cursor_by_array(index_subset_array)
            degenerate_array[ii] = cursor

        return degenerate_array

    def fill_up_appropriate_crude_index_subset(
        self,
        index_of_degeneration: np.ndarray,
        position: int,
        lb: int,
        rb: int,
        crude_index_array: List[int],
        virtual_residue_index: int,
        index_subset: List[int]
    ):
        size = len(crude_index_array)
        window_size = rb - lb + 1
        start = position + lb
        end = position + rb + 1

        for k, i in enumerate(range(start, end)):
            if 0 <= i < size:
                index_subset[k] = index_of_degeneration[k][crude_index_array[i]]
            else:
                index_subset[k] = index_of_degeneration[k][virtual_residue_index]

    def refresh_sequence_and_stuff(self, sequence: str):
        """
        Унифицированный метод, аналогичный FrequencyExtrapolation.
        Сохраняет массив дегенеративных индексов в self.processed_index.
        """
        self.processed_index = self.translate_sequence_to_degenerate_array(sequence)
    def translate_cursor_to_sequence(self, degenerate_cursor: int) -> List[str]:
        """Перевод курсора дегенерации в последовательность."""
        redundant_sequence = []
        current_array = self.curly_calculus_system.get_array_by_cursor(degenerate_cursor)
        for kk, current_index in enumerate(current_array):
            current_word = self.degeneration[kk][current_index]
            redundant_sequence.append(current_word)
        return redundant_sequence



if __name__ == "__main__":

  DSP_name="_PB_W3_trivial"
  path_to_frequency_store="../FrequencyExtrapolation/" 
  
  generator = DegeneratePredictorGenerator(DSP_name,path_to_frequency_store)
  
  sequence = "AAAAAARAPRARAKALRLLLKLLKLLSRYWVRVKRLLL"
  predictors = generator.translate_sequence_to_degenerate_array(sequence)

  generator.refresh_sequence_and_stuff(sequence)
  print("Длина:", len(generator.processed_index), "==", len(sequence))
  print("Первые предикторы:", generator.processed_index)

  processed_index=generator.processed_index
  print("processed_index:")
  print(processed_index)

  for i in range(len(sequence)):
    word_set=generator.translate_cursor_to_sequence(processed_index[i])
    print(f'{i}: {word_set}->{predictors[i]}')

  print(f'number_of_elements:{generator.number_of_elements}')

