import sys
# Это словать всегда такой поэтому прибит гвоздиком  FIX если кластеры будут другие
PB_to_index = {               #
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15
}

def PB_word_to_indexes( letters):
        result = []
        for ch in letters:
            if ch not in PB_to_index:
                # Сообщаем о проблеме и завершаем программу
                print(f"Неизвестный символ: {ch!r}. в скрипте PB_word_to_indexes(). Программа прерывается.")
                sys.exit(1)
            result.append(PB_to_index[ch])
        return result
