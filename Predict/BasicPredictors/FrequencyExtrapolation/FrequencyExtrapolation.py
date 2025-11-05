import numpy as np
import struct
from pathlib import Path
import logging
from typing import List, Dict, Optional

from Scheduler import Scheduler
from CurlyCalculusSystem import CurlyCalculusSystem
from AminoAcidConverter import AminoAcidConverter
from aa_sequence_to_index_array import aa_sequence_to_index_array

logging.basicConfig(level=logging.INFO)

class FrequencyExtrapolation:
    def __init__(self, name: str, path_to_frequency_store: str):
        self.name: str = name
        self.path_to_frequency_store: Path = Path(path_to_frequency_store)

        self.degeneration: List[List[str]] = []
        self.position_shift: List[int] = []
        self.number_of_elements: int = 0
        self.number_of_classes: int = 0
        self.fragment_length: int = 0
        self.cluster_motif_coordinates: Optional[np.ndarray] = None
        self.distance_record: Dict[int, List[float]] = {}
        self.index_to_nonzero_occurrence: Dict[int, int] = {}
        self.buffer_array: np.ndarray = np.array([])
        self.large_buffer_array: np.ndarray = np.array([])

        self.shift_to_index_to_pos_start: int = 0
        self.record_length: int = 0
        # Файл больше не хранится как поток
        self.standby_datastream = None

        self.window_size = 0
        self.left_border_value = None
        self.right_border_value = None
        self.curly_calculus_system = None

        self.distance_to_clusters_sum: List[List[float]] = []
        self.squared_distance_to_clusters_sum: List[List[float]] = []
        self.inverse_distance_to_clusters_sum: List[List[float]] = []
        self.inverse_squared_distance_to_clusters_sum: List[List[float]] = []

        self.tot_distance_to_clusters_sum: List[float] = []
        self.tot_squared_distance_to_clusters_sum: List[float] = []
        self.tot_inverse_distance_to_clusters_sum: List[float] = []
        self.tot_inverse_squared_distance_to_clusters_sum: List[float] = []

        self.respect_occurence: List[int] = []
        self.total_sample_size: int = 0
        self.processed_index: List[int] = []

        # --- Настройка scheduler, длины фрагмента, числа классов ---
        scheduler = Scheduler(f"{self.path_to_frequency_store}/{name}/sheduler")
        self.fragment_length = int(scheduler.option_meaning('FRAGNMENT_LENGTH'))
        self.number_of_classes = int(scheduler.option_meaning('NUMBER_OF_CLASSES'))

        # --- Считываем degeneration, window и переводим в индексы ---
        self.read_degeneration(str(self.path_to_frequency_store / name / "current.degeneration"))
        self.assign_window_size_left_right_border_value()
        self.translate_degeneration_to_index()

        # --- Инициализация CurlyCalculusSystem и числа элементов ---
        self.init_curly_calculus_system()
        self.number_of_elements = self.curly_calculus_system.get_number_of_elements()

        self.number_of_items_in_record = 4 * self.number_of_classes
        self.record_length = self.number_of_items_in_record * 4

        # --- Читаем файл целиком в память ---
        file_path = Path(f"{self.path_to_frequency_store}/{name}/base/together.freq_data")
        if file_path.is_file():
            with file_path.open('rb') as f:
                self._buffer: bytes = f.read()   # <<-- весь файл в память
        else:
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        self.init_essential_indexes()

        # --- Читаем сразу итоговые блоки из буфера (tot_* методы уже используют buffer) ---
        self.tot_distance_to_clusters_sum = self.get_tot_distance_to_clusters_sum()
        self.tot_squared_distance_to_clusters_sum = self.get_tot_squared_distance_to_clusters_sum()
        self.tot_inverse_distance_to_clusters_sum = self.get_tot_inverse_distance_to_clusters_sum()
        self.tot_inverse_squared_distance_to_clusters_sum = self.get_tot_inverse_squared_distance_to_clusters_sum()
        self.total_sample_size = self.get_total_sample_size()


    # ====== ЧТЕНИЕ ИЗ БУФЕРА: вспомогательные методы ======

    def _read_int(self, offset: int) -> int:
        """Считывает 4-байтный little-endian int из буфера."""
        return struct.unpack_from('<i', self._buffer, offset)[0]

    def _read_floats(self, offset: int, count: int) -> List[float]:
        """Считывает count float-ов (4-байт каждый little-endian) из буфера."""
        tpl = struct.unpack_from(f'<{count}f', self._buffer, offset)
        return list(tpl)


    # ====== ИНИЦИАЛИЗАЦИЯ ИНДИКСОВ ======

    def init_essential_indexes(self):
        """Инициализация базовых индексов и смещений для работы с буфером."""
        # Читаем non_zero_occurrence_number из первых 4 байт
        self.non_zero_occurrence_number = self._read_int(0)

        # Вычисляем сдвиг для массива индексов non-zero:
        # 4 (int) + non_zero_occurrence_number * (record_length + 8)
        self.shift_to_index_to_pos_start = (
            4 + 
            self.non_zero_occurrence_number * (self.record_length + 8)
        )


    # ====== ПУБЛИЧНЫЕ МЕТОДЫ ======

    def get_occurrence(self, record_index: int) -> int:
        if record_index < 0 or record_index >= self.number_of_elements:
            return -1

        # Считаем non_zero_occurrence_record_index
        total_shift = self.shift_to_index_to_pos_start + record_index * 4
        non_zero_occurrence_record_index = self._read_int(total_shift)

        if non_zero_occurrence_record_index == -1:
            return 0

        # Считаем непосредственно occurrence (4 байта)
        shift = 4 + non_zero_occurrence_record_index * (self.record_length + 8) + 4
        current_occurrence = self._read_int(shift)
        return current_occurrence


    def get_single_distance_to_clusters_sum(self, record_index: int) -> List[float]:
        record = [0.0] * self.number_of_classes
        if record_index < 0 or record_index >= self.number_of_elements:
            return record

        if self.get_occurrence(record_index) == 0:
            return record

        total_shift = self.shift_to_index_to_pos_start + record_index * 4
        non_zero_occurrence_record_index = self._read_int(total_shift)

        shift = 4 + non_zero_occurrence_record_index * (self.record_length + 8) + 8
        return self._read_floats(shift, self.number_of_classes)


    def get_single_squared_distance_to_clusters_sum(self, record_index: int) -> List[float]:
        record = [0.0] * self.number_of_classes
        if record_index < 0 or record_index >= self.number_of_elements:
            return record

        if self.get_occurrence(record_index) == 0:
            return record

        total_shift = self.shift_to_index_to_pos_start + record_index * 4
        non_zero_occurrence_record_index = self._read_int(total_shift)

        shift = 4 + non_zero_occurrence_record_index * (self.record_length + 8) + 8 + self.number_of_classes * 4
        return self._read_floats(shift, self.number_of_classes)


    def get_single_inverse_distance_to_clusters_sum(self, record_index: int) -> List[float]:
        record = [0.0] * self.number_of_classes
        if record_index < 0 or record_index >= self.number_of_elements:
            return record

        if self.get_occurrence(record_index) == 0:
            return record

        total_shift = self.shift_to_index_to_pos_start + record_index * 4
        non_zero_occurrence_record_index = self._read_int(total_shift)

        shift = 4 + non_zero_occurrence_record_index * (self.record_length + 8) + 8 + 2 * self.number_of_classes * 4
        return self._read_floats(shift, self.number_of_classes)


    def get_single_inverse_squared_distance_to_clusters_sum(self, record_index: int) -> List[float]:
        record = [0.0] * self.number_of_classes
        if record_index < 0 or record_index >= self.number_of_elements:
            return record

        if self.get_occurrence(record_index) == 0:
            return record

        total_shift = self.shift_to_index_to_pos_start + record_index * 4
        non_zero_occurrence_record_index = self._read_int(total_shift)

        shift = 4 + non_zero_occurrence_record_index * (self.record_length + 8) + 8 + 3 * self.number_of_classes * 4
        return self._read_floats(shift, self.number_of_classes)


    def get_tot_distance_to_clusters_sum(self) -> List[float]:
        shift = (
            4
            + self.non_zero_occurrence_number * (self.record_length + 2 * 4)
            + self.number_of_elements * 4
        )
        return self._read_floats(shift, self.number_of_classes)


    def get_tot_squared_distance_to_clusters_sum(self) -> List[float]:
        shift = (
            4
            + self.non_zero_occurrence_number * (self.record_length + 2 * 4)
            + self.number_of_elements * 4
            + self.number_of_classes * 4
        )
        return self._read_floats(shift, self.number_of_classes)


    def get_tot_inverse_distance_to_clusters_sum(self) -> List[float]:
        shift = (
            4
            + self.non_zero_occurrence_number * (self.record_length + 2 * 4)
            + self.number_of_elements * 4
            + 2 * self.number_of_classes * 4
        )
        return self._read_floats(shift, self.number_of_classes)


    def get_tot_inverse_squared_distance_to_clusters_sum(self) -> List[float]:
        shift = (
            4
            + self.non_zero_occurrence_number * (self.record_length + 2 * 4)
            + self.number_of_elements * 4
            + 3 * self.number_of_classes * 4
        )
        return self._read_floats(shift, self.number_of_classes)


    def get_total_sample_size(self) -> int:
        shift = (
            4
            + self.non_zero_occurrence_number * (self.record_length + 2 * 4)
            + self.number_of_elements * 4
            + 4 * self.number_of_classes * 4
        )
        return self._read_int(shift)


    # ====== Прочие методы (без изменений или с небольшими поправками) ======

    def refresh_sequence_and_stuff(self, sequence: str):
        self.distance_to_clusters_sum.clear()
        self.squared_distance_to_clusters_sum.clear()
        self.inverse_distance_to_clusters_sum.clear()
        self.inverse_squared_distance_to_clusters_sum.clear()
        self.respect_occurence.clear()
        self.processed_index.clear()

        self.processed_index = self.translate_sequence_to_degenerate_array(sequence)

        for idx in self.processed_index:
            self.distance_to_clusters_sum.append(self.get_single_distance_to_clusters_sum(idx))
            self.squared_distance_to_clusters_sum.append(self.get_single_squared_distance_to_clusters_sum(idx))
            self.inverse_distance_to_clusters_sum.append(self.get_single_inverse_distance_to_clusters_sum(idx))
            self.inverse_squared_distance_to_clusters_sum.append(self.get_single_inverse_squared_distance_to_clusters_sum(idx))
            self.respect_occurence.append(self.get_occurrence(idx))


    def translate_degeneration_to_index(self):
        map_size = AminoAcidConverter.get_size_aminoacid_set()
        degeneration_size = len(self.degeneration)
        self.index_of_degeneration = np.full((degeneration_size, map_size), -1, dtype=int)

        for ii, degeneration_group in enumerate(self.degeneration):
            for jj, word in enumerate(degeneration_group):
                for aa in word:
                    aa_index = AminoAcidConverter.aminoacid_to_index(aa)
                    self.index_of_degeneration[ii, aa_index] = jj

    def init_curly_calculus_system(self):
        bases = [len(degen) for degen in self.degeneration]
        self.curly_calculus_system = CurlyCalculusSystem(bases)

    def assign_window_size_left_right_border_value(self):
        self.window_size = len(self.position_shift)
        self.left_border_value = self.position_shift[0] if self.position_shift else None
        self.right_border_value = self.position_shift[-1] if self.position_shift else None

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

        assert len(index_subset) == window_size, "index_subset size mismatch"

        kk = 0
        for ii in range(start, end):
            if 0 <= ii < size:
                index_subset[kk] = index_of_degeneration[kk][crude_index_array[ii]]
            else:
                index_subset[kk] = index_of_degeneration[kk][virtual_residue_index]
            kk += 1

    def get_degeneration(self) -> List[List[str]]:
        return self.degeneration

    def get_respect_occurence(self) -> List[int]:
        return self.respect_occurence

    def get_fragment_length(self) -> int:
        return self.fragment_length

    def get_processed_index(self) -> List[int]:
        return self.processed_index

    def get_number_of_elements(self) -> int:
        return self.number_of_elements

    def get_cluster_motif_coordinates(self) -> Optional[np.ndarray]:
        return self.cluster_motif_coordinates

    def get_name(self) -> str:
        return self.name

    def calc_cluster_mutual_distance(self) -> np.ndarray:
        return np.random.rand(5, 5)

    def i_freq_data_stream(self, base_file_name: str):
        path = Path(base_file_name)
        if not path.exists():
            logging.error(f"Файл {base_file_name} не найден.")
            return
        with path.open('r') as file:
            data = file.readlines()
            logging.info(f"Прочитано {len(data)} строк из {base_file_name}")

    def o_freq_data_stream(self, base_file_name: str):
        with Path(base_file_name).open('w') as file:
            file.write("# Frequency Extrapolation Data\nSample data\n")

    def prepare_together_freq_data(self):
        pass

    def init_cluster_motif(self):
        self.cluster_motif_coordinates = np.random.rand(10, 3)

    def read_degeneration(self, degeneration_file_name: str):
        path = Path(degeneration_file_name)
        if not path.exists():
            raise FileNotFoundError(f"Can't find degeneration file: {degeneration_file_name}")

        with path.open('r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(('/', '#')):
                    continue
                parts = line.split()
                if parts:
                    dummy_int = int(parts[0])
                    self.position_shift.append(dummy_int)
                    self.degeneration.append(parts[1:])

    def translate_cursor_to_sequence(self, degenerate_cursor: int) -> List[str]:
        redundant_sequence = []
        current_array = self.curly_calculus_system.get_array_by_cursor(degenerate_cursor)
        for kk, current_index in enumerate(current_array):
            current_word = self.degeneration[kk][current_index]
            redundant_sequence.append(current_word)
        return redundant_sequence
