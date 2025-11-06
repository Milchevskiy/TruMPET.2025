def read_aaindex_mutant3(file_path, log_path="missing_values_log.txt"):
    """
    Читает файл aaindex_mutant3.txt и возвращает словарь:
    aaindex_dict[свойство][аминокислота] = значение (float или None).
    Все пропущенные значения ('-') логируются в указанный файл.
    """
    aaindex_dict = {}
    missing_entries = []

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Первая строка — названия аминокислот
    header = lines[0].split()
    if header[0].upper() != "PROPERTY":
        raise ValueError("Неверный формат: первая строка должна начинаться с 'Property'")
    amino_acids = header[1:]

    # Обработка остальных строк
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split()
        prop_name = parts[0]
        raw_values = parts[1:]

        if len(raw_values) != len(amino_acids):
            raise ValueError(f"Строка {line_number}: свойство {prop_name} — число значений не совпадает с числом аминокислот")

        property_dict = {}
        for aa, val in zip(amino_acids, raw_values):
            if val == '-':
                missing_entries.append((line_number, prop_name, aa))
                property_dict[aa] = None
            else:
                try:
                    property_dict[aa] = float(val)
                except ValueError:
                    raise ValueError(f"Неверное значение '{val}' в свойстве '{prop_name}', аминокислота '{aa}', строка {line_number}")

        aaindex_dict[prop_name] = property_dict

    # Логирование пропущенных значений в файл
#    if missing_entries:
#        with open(log_path, "w") as log:
#            for line_number, prop, aa in missing_entries:
#                log.write(f"row {line_number}: property '{prop}', aminoacid '{aa}'\n")

#        print(f"\n⚠️  Пропущенные значения ('-') сохранены в: {log_path}")

    return aaindex_dict

if __name__ == '__main__':
  # Путь к файлу
  file_path = "aaindex.data/aaindex_mutant3.txt"

  # Чтение словаря
  aaindex_dict = read_aaindex_mutant3(file_path)

  # Извлечение свойства ARGP820101 для аминокислоты CYS
  value = aaindex_dict["ARGP820101"]["CYS"]
  print(f"Значение ARGP820101 для CYS: {value}")

  value = aaindex_dict["ARGP820101"]["HYP"]
  print(f"Значение ARGP820101 для HYP: {value}")

  value = aaindex_dict["ARGP820101"]["PRO"]
  print(f"Значение ARGP820101 для PRO: {value}")


