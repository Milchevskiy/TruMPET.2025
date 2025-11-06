import os

def read_amino_acid_file(filepath):
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    # Проверка на минимальное количество строк
    if len(lines) < 3:
        raise ValueError(f"Недостаточно строк в файле {filepath} (получено {len(lines)})")

    aa3 = lines[0].split()
    aa1 = ''.join(lines[1].split())
    dssp = ''.join(lines[2].split())
    return aa3, aa1, dssp

def iter_RawData(file_list, path_to_data_store,path_to_Datasets_store):
    file_list_path = os.path.join(path_to_Datasets_store, file_list)

    with open(file_list_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]

    for relative_path in file_paths:
        relative_path = relative_path.upper()
        path = os.path.join(path_to_data_store, relative_path)
        path = path + ".aa3_aa1_ss"
        if not os.path.exists(path):
            print(f"Файл не найден: {path}")
            continue

        try:
            aa3, aa1, dssp = read_amino_acid_file(path)

            if not aa3 or not aa1 or not dssp:
                print(f"Пропуск пустого файла: {path}")
                continue

            if all(c == '*' for c in dssp):
                print(f"Все символы DSSP равны '*', файл пропущен: {path}")
                continue

            if all(c == 'X' for c in aa1):
                print(f"Все символы аминокислот — 'X', файл пропущен: {path}")
                continue

            if len(aa3) != len(aa1) or len(aa1) != len(dssp):
                print(f"Несовпадение длин в файле {path}: aa3={len(aa3)}, aa1={len(aa1)}, dssp={len(dssp)}")
                continue

            yield (path, aa3, aa1, dssp)

        except Exception as e:
            print(f"Ошибка при обработке файла {path}: {e}")

if __name__ == "__main__":
    file_list               = '___tiny.txt'
    path_to_data_store      = '../DATA/aa3_aa1_ss/'
    path_to_Datasets_store  = '../DATA/DataSets/'

    for filename, aa3, aa1, dssp in iter_RawData(file_list, path_to_data_store,path_to_Datasets_store):
        print(f"\nФайл: {filename}")
        print("3-буквенные:", aa3)
        print("1-буквенные:", aa1)
        print("DSSP:", dssp)

