from typing import List

class AminoAcidConverter:
    _aa_to_index = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'O': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'X': 20, 'Y': 21
    }
    
    _index_to_aa = {v: k for k, v in _aa_to_index.items()}
    
    @classmethod
    def aminoacid_to_index(cls, aa: str) -> int:
        """Преобразует аминокислоту в индекс, возвращает -1, если не найдено."""
        return cls._aa_to_index.get(aa, -1)
    
    @classmethod
    def index_to_aminoacid(cls, index: int) -> str:
        """Преобразует индекс в аминокислоту, возвращает '-' если не найдено."""
        return cls._index_to_aa.get(index, '-')
    
    @classmethod
    def get_size_aminoacid_set(cls) -> int:
        """Возвращает размер множества аминокислот."""
        return len(cls._aa_to_index)
    
    @classmethod
    def is_standard_aa(cls, aa: str) -> bool:
        """Проверяет, является ли аминокислота стандартной."""
        return aa in cls._aa_to_index
    
    @classmethod
    def get_virtual_residue_index(cls) -> int:
        """Возвращает индекс виртуального остатка 'O'."""
        return cls.aminoacid_to_index('O')


def aa_sequence_to_index_array(sequence: str) -> list:
    """Преобразует последовательность аминокислот в массив индексов."""
    return [AminoAcidConverter.aminoacid_to_index(aa) for aa in sequence]


if __name__ == "__main__":
    # Пример использования
    sequence = "RAPRARAKALRLLLKLLKLLSRYWVRVKRLLL"  # пример последовательности аминокислот
    index_array = aa_sequence_to_index_array(sequence)
    
    print("Последовательность:", sequence)
    print("Массив индексов:", index_array)
