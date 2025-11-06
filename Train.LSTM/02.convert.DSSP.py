#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from parse_loop import parse_loop

# ——————————————————————————————————————————————
PATH_TO_CIF_DSSP_STORE = "./DSSP/"  # where downloaded MMCIF-dssp files are stored
PATH_TO_RESULT_STORE   = "Databases/AA3_AA1_SS/"
# ——————————————————————————————————————————————

if len(sys.argv) != 2:
    print("Usage: python3 02.convert.DSSP.py chainlist.txt")
    exit(-1)

input_file = sys.argv[1]

# Mapping from three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def map_seq_to_ss(res_scheme, res_dssp):
    """
    Возвращает список кортежей (трёхбуквенный код, SS) в порядке res_scheme.
    Если pdb_seq_num=='?' или номер не найден в DSSP — SS='*'.
    """
    mon_ids      = res_scheme["_pdbx_poly_seq_scheme.mon_id"]
    ndb_seq_num  = res_scheme["_pdbx_poly_seq_scheme.ndb_seq_num"]

    label_seq_ids = res_dssp["_dssp_struct_summary.label_seq_id"]
    ss_values     = res_dssp["_dssp_struct_summary.secondary_structure"]

    ss_map = {
        label_seq_ids[i]: ss_values[i]
        for i in range(len(label_seq_ids))
    }

    result = []
    for mon, seq in zip(mon_ids, ndb_seq_num):
        if seq != "?":
            ss = ss_map.get(seq, "*")
        else:
            ss = "*"
        result.append((mon, ss))
    return result

def extract_chain_sequence_and_ss(path_to_dssp_cif: str,
                                  chain_id: str,
                                  path_to_result_file: str) -> None:
    """
    Читает CIF+DSSP файл, вытаскивает заданную цепь chain_id и пишет в файл:
      1) трёхбуквенная последовательность (через пробел)
      2) однобуквенная (нестандартным остаткам — 'X')
      3) SS-последовательность ('*' для неизвестных)
    """
    # Считываем всё
    with open(path_to_dssp_cif, encoding='utf-8') as f:
        lines = f.readlines()

    # Парсим полипептидную схему для цепи
    scheme = parse_loop(
        lines,
        first_loop_item_name="_pdbx_poly_seq_scheme.asym_id",
        filter_field="_pdbx_poly_seq_scheme.pdb_strand_id",
        filter_value=chain_id
    )
    if not scheme.get("_pdbx_poly_seq_scheme.mon_id"):
        raise ValueError(f"Цепь '{chain_id}' не найдена в схеме.")

    # Метка asym_id для DSSP
    label_asym = scheme["_pdbx_poly_seq_scheme.asym_id"][0]

    # Парсим DSSP для той же метки
    dssp = parse_loop(
        lines,
        first_loop_item_name="_dssp_struct_summary.entry_id",
        filter_field="_dssp_struct_summary.label_asym_id",
        filter_value=label_asym
    )

    # Маппим остатки на SS
    seq_ss = map_seq_to_ss(scheme, dssp)

    # Формируем строки для записи
    three_letter = [mon for mon, ss in seq_ss]
    one_letter   = [THREE_TO_ONE.get(mon, 'X') for mon in three_letter]
    ss_seq       = [ss for mon, ss in seq_ss]

    line1 = ' '.join(three_letter)
    line2 = ''.join(one_letter)
    line3 = ''.join(ss_seq)

    # Записываем три строки в файл
    with open(path_to_result_file, 'w', encoding='utf-8') as out:
        out.write(line1 + '\n')
        out.write(line2 + '\n')
        out.write(line3 + '\n')

def main():
    # Создаём папку для результатов, если нужно
    os.makedirs(PATH_TO_RESULT_STORE, exist_ok=True)

    # Читаем файл списка PDB+chain
    try:
        flist = open(input_file, encoding='utf-8')
    except IOError as e:
        print(f"Не удалось открыть список '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    with flist:
        for raw in flist:
            entry = raw.strip()
            if not entry:
                continue
            pdb_id   = entry[:4]
            pdb_id   = pdb_id.lower()

            chain_id = entry[4:]
            cif_file = os.path.join(PATH_TO_CIF_DSSP_STORE, f"{pdb_id}.cif")

            pdb_id   = pdb_id.upper()

            result_file = os.path.join(PATH_TO_RESULT_STORE, f"{pdb_id}{chain_id}.aa3_aa1_ss")

            if not os.path.exists(cif_file):
                print(f"[SKIP] CIF-файл не найден: {cif_file}", file=sys.stderr)
                continue

            try:
                extract_chain_sequence_and_ss(cif_file, chain_id, result_file)
            except Exception as e:
                print(f"[ERROR] {pdb_id}{chain_id}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
