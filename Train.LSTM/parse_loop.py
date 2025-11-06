def parse_loop(lines,
               first_loop_item_name,
               filter_field=None,
               filter_value=None):
    """
    Парсит блок данных, начинающийся с тега first_loop_item_name.
    Собирает подряд все _теги_ (пока они идут), затем строки данных
    до первой пустой или начинающейся с '#'.

    Если заданы filter_field и filter_value, отбирает только строки,
    где колонка filter_field == filter_value.
    """
    fields = []
    data = {}
    it = iter(lines)

    # 1) Найти первый тэг полей
    for line in it:
        if line.strip() == first_loop_item_name:
            fields.append(first_loop_item_name)
            break
    else:
        raise ValueError(f"Не найден тэг {first_loop_item_name!r}")

    # 2) Собирать все следующие тэги (строки, начинающиеся с '_')
    for line in it:
        tag = line.strip()
        if tag.startswith("_"):
            fields.append(tag)
        else:
            # первая строка данных
            first_data = line
            break

    if not fields:
        raise ValueError(f"Не найдено ни одного поля после {first_loop_item_name}")

    # 3) Проверка фильтра
    if filter_field:
        if filter_field not in fields:
            raise ValueError(f"Поле фильтра {filter_field!r} не найдено среди {fields}")
        idx_filter = fields.index(filter_field)
    else:
        idx_filter = None

    # 4) Инициализируем пустые списки под все поля
    data = {fld: [] for fld in fields}

    # 5) Функция обработки одной строки данных
    def process_row(row):
        cols = row.split()
        if len(cols) != len(fields):
            raise ValueError(f"Ожидалось {len(fields)} колонок, а получили {len(cols)}: {cols}")
        # проверка фильтра
        if idx_filter is not None and cols[idx_filter] != filter_value:
            return
        for fld, val in zip(fields, cols):
            data[fld].append(val)

    # 6) Обработать первую строку данных, если она не комментарий
    if first_data and not first_data.lstrip().startswith("#"):
        process_row(first_data)

    # 7) Дальше все строки до пустой или '#'   
    for line in it:
        s = line.strip()
        if not s or s.startswith("#"):
            break
        process_row(line)

    return data

if __name__ == "__main__":

  with open("7af2.cif_dssp", encoding="utf-8") as f:
    lines = f.readlines()

  res = parse_loop(
    lines,
    first_loop_item_name="_pdbx_poly_seq_scheme.asym_id",
    filter_field="_pdbx_poly_seq_scheme.pdb_strand_id",
    filter_value="AAA"
  )

  print(res["_pdbx_poly_seq_scheme.mon_id"])
  # и все остальные поля появятся в res по ключу их названия


