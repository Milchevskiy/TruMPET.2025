import sys
import os
import logging

# Добавляем путь до модуля FrequencyExtrapolation (если не в sys.path)
sys.path.append(os.path.join(os.path.dirname(__file__), 'FrequencyExtrapolation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'aaindex'))

from FrequencyExtrapolation import FrequencyExtrapolation
from Scheduler import Scheduler
from CurlyCalculusSystem import CurlyCalculusSystem
from AminoAcidConverter import AminoAcidConverter
from load_aaindex_data import load_aaindex_data
from read_aaindex_mutant3 import read_aaindex_mutant3

from DegeneratePredictorGenerator import DegeneratePredictorGenerator

class _CommonConstants:
    def __init__(self,
                 path_to_frequency_store: str,
                 path_to_aaindex_file: str,
                 path_to_aaindex_tri_letter_file: str,
                 log_level: int = logging.INFO,
                 log_to_file: bool = False,
                 log_file_name: str = "log_output.txt"):

         self.frequency_map_dict = {}
         self.DSP_map_dict       = {} 
         self.aaindex_data = {}
         self.aaindex_data_tri_letter = {} 
         self.path_to_frequency_store = path_to_frequency_store
         self.path_to_aaindex_file = path_to_aaindex_file

         self.logger = logging.getLogger("CommonLogger")
         self.logger.setLevel(logging.DEBUG)          # общий уровень логгера (можно оставить DEBUG)
         formatter = logging.Formatter("[%(levelname)s] %(message)s")

        # не плодим хендлеры
         if self.logger.hasHandlers():
          self.logger.handlers.clear()

       # >>> Консоль: показываем только WARNING и выше
         console_handler = logging.StreamHandler()
         console_handler.setLevel(logging.WARNING)    # <-- ключевая строка
         console_handler.setFormatter(formatter)
         self.logger.addHandler(console_handler)

       # >>> Файл: если нужен лог INFO — оставляем уровень INFO
         if log_to_file:
           file_handler = logging.FileHandler(log_file_name, encoding="utf-8")
           file_handler.setLevel(logging.INFO)      # файл пишет INFO и выше
           file_handler.setFormatter(formatter)
           self.logger.addHandler(file_handler)

       # не отдаём записи в root-логгер (иначе появится формат "INFO:CommonLogger: ...")
         self.logger.propagate = False                # <-- ключевая строка


         self.logger.debug(f"CommonConstants инициализирован с frequency_store='{self.path_to_frequency_store}', aaindex_store='{self.path_to_aaindex_file}'")

# 📌 load aaindex
         self.aaindex_data = load_aaindex_data(path_to_aaindex_file)
         if self.aaindex_data:
          self.logger.info(f"Данные aaindex успешно прочитаны, найдено {len(self.aaindex_data)} записей.")
         else:
          self.logger.error("Не удалось прочитать данные aaindex!")


         self.aaindex_data_tri_letter = read_aaindex_mutant3(path_to_aaindex_tri_letter_file)
         if self.aaindex_data_tri_letter:
          self.logger.info(f"Данные aaindex успешно прочитаны, найдено {len(self.aaindex_data_tri_letter)} записей.")
         else:
          self.logger.error("Не удалось прочитать данные aaindex_mutant3!")
        

#    def get_aaindex_data(self, position_in_chain: int) -> float:

    def get_aaindex_data(self) -> dict:
        """
        Возвращает словарь данных aaindex.
        """
        return self.aaindex_data

    def get_aaindex_data_tri_letter(self) -> dict:
        """
        Возвращает словарь данных aaindex.
        """
        return self.aaindex_data_tri_letter


    def add_frequency_item(self, frequency_name: str) -> bool:
        if frequency_name in self.frequency_map_dict:
            return True
            #self.logger.warning(f"Frequency map '{frequency_name}' уже существует.")
        else:
          try:
            freq_extrap = FrequencyExtrapolation(frequency_name, self.path_to_frequency_store)
            self.frequency_map_dict[frequency_name] = freq_extrap
            self.logger.info(f"FrequencyExtrapolation '{frequency_name}' успешно добавлен.")
            return True
          except Exception as e:
            self.logger.error(f"Ошибка при создании FrequencyExtrapolation для '{frequency_name}': {e}")
            return False

    def remove_frequency_item(self, frequency_name: str) -> bool:
        """
        Удаляет объект FrequencyExtrapolation с ключом frequency_name из словаря.
        Возвращает True, если удаление прошло успешно, и False — если элемента не было.
        """
        if frequency_name not in self.frequency_map_dict:
            self.logger.warning(f"Frequency map '{frequency_name}' не найден в словаре и не может быть удалён.")
            return False

        try:
            # Если внутри FrequencyExtrapolation есть ресурсы, которые нужно освободить (например, файлы, открытые потоки),
            # можно вызвать здесь какой-то cleanup-метод. Например:
            # self.frequency_map_dict[frequency_name].close()  # если он есть

            del self.frequency_map_dict[frequency_name]
            self.logger.info(f"FrequencyExtrapolation '{frequency_name}' успешно удалён.")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при удалении FrequencyExtrapolation '{frequency_name}': {e}")
            return False


    def add_DSP_item(self, DSP_name: str) -> bool:
        if DSP_name in self.DSP_map_dict:
            return True  # полностью тихо, без warning/print
            #self.logger.warning(f"DSP  map '{DSP_name}' уже существует.")
        else:
          try:
            DSP_item = DegeneratePredictorGenerator(DSP_name, self.path_to_frequency_store)
            self.DSP_map_dict[DSP_name] = DSP_item
            self.logger.info(f"DSP_item '{DSP_name}' успешно добавлен.")
            return True
          except Exception as e:
            self.logger.error(f"Ошибка при создании DSP_item для '{DSP_name}': {e}")
            return False


if __name__ == "__main__":

    path_to_frequency_store="../DATA/FrequencyExtrapolation/"
    path_to_aaindex_file="../DATA/aaindex/aaindex.data/aaindex.txt" 
    path_to_aaindex_tri_letter_file = "../DATA/aaindex/aaindex.data/aaindex_mutant3.txt"

    common = _CommonConstants(
                      path_to_frequency_store,
                      path_to_aaindex_file,
                      path_to_aaindex_tri_letter_file,
                      log_level=logging.DEBUG)

    aaindex_data_tri_letter=common.get_aaindex_data_tri_letter()    

    value = aaindex_data_tri_letter["ARGP820101"]["CYS"]
    print(f"Значение ARGP820101 для CYS: {value}")

    value = aaindex_data_tri_letter["ARGP820101"]["HYP"]
    print(f"Значение ARGP820101 для HYP: {value}")

    value = aaindex_data_tri_letter["ARGP820101"]["PRO"]
    print(f"Значение ARGP820101 для PRO: {value}")



    aaindex_data=common.get_aaindex_data()    

    value = aaindex_data["ARGP820101"]["P"]
    print(f"Значение ARGP820101 для P: {value}")


    if common.add_frequency_item("PB_W7_tail_GP"):
        freq_extrap = common.frequency_map_dict["PB_W7_tail_GP"]
        occurence_0 = freq_extrap.get_occurrence(0)
        common.logger.info(f"occurence_0: {occurence_0}")
    else:
        common.logger.error("Не удалось загрузить карту PB_W7_tail_GP")



    if common.add_DSP_item("_PB_W3_trivial"):
        DSP_item = common.DSP_map_dict["_PB_W3_trivial"]
#        occurence_0 = freq_extrap.get_occurrence(0)
#        common.logger.info(f"occurence_0: {occurence_0}")
    else:
        common.logger.error("Не удалось загрузить карту _PB_W3_trivial")


