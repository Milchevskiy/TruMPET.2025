def load_aaindex_data(filename):
    """
    Читает файл с данными AAINDEX и сохраняет их в словарь.
    
    :param filename: Путь к файлу
    :return: Словарь {свойство: {аминокислота: значение, ...}, ...}
    """
    amino_acids = 'ACDEFGHIKLMNOPQRSTVWXY'  # Порядок аминокислот
    data = {}  # Итоговая структура

    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()  # Разбиваем строку по пробелам
            if len(parts) != 23:  # 1 название + 22 числа
                continue  # Пропускаем строки с ошибками

            property_name = parts[0]  # Название свойства
            values = list(map(float, parts[1:]))  # Числовые значения

            # Создаем словарь {аминокислота: значение}
            data[property_name] = {aa: val for aa, val in zip(amino_acids, values)}

    return data

if __name__ == '__main__':

  # 📌 Пример использования:
  aaindex_data = load_aaindex_data("../aaindex.data/aaindex.txt")


  property_name = 'ANDN920101'  # Например, гидрофобность
#amino_acid = 'A'  # Аланин

  amino_acids = 'ACDEFGHIKLMNOPQRSTVWXY'  # Порядок аминокислот

  for amino_acid in amino_acids:
    value = aaindex_data.get(property_name, {}).get(amino_acid, None)
    print(f"Значение {property_name} для {amino_acid}: {value}")

