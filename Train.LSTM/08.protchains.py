#!/usr/bin/env python3
import sys
import os
import logging
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'BasicPredictors/'))

from _CommonConstants import _CommonConstants
from _create_predictors_set_common import _create_predictors_set_common
from create_predictors_MLP import create_predictors_MLP
from encode_dssp_sequence import encode_dssp_sequence
from iter_RawData import iter_RawData

def LSTM_make_data(
    file_list,
    path_to_Datasets_store,
    path_to_data_store,
    path_to_frequency_store,
    path_to_aaindex_file,
    path_to_aaindex_tri_letter_file,
    dssp_to_index,
    task_set_file,
    output_dir="output.lda"
):

    logging.basicConfig(filename=os.path.join(output_dir, "data_generation.log"),
                        filemode='w',
                        level=logging.INFO)

    common = _CommonConstants(
        path_to_frequency_store,
        path_to_aaindex_file,
        path_to_aaindex_tri_letter_file,
        log_level=logging.WARNING
    )
    objects = _create_predictors_set_common(task_set_file, common)

    dataset_ids = []

    data_iter = list(iter_RawData(file_list, path_to_data_store,path_to_Datasets_store))

    for filename, aa3, aa1, dssp in tqdm(data_iter, desc="Processing sequences"):
        name_only = os.path.basename(filename)
        base, ext = os.path.splitext(name_only)
        while ext:
            base, ext = os.path.splitext(base)

        X_phys = create_predictors_MLP(objects, aa1, aa3, common)

        X = X_phys
        y = encode_dssp_sequence(dssp, dssp_to_index)

        if X.shape[0] != len(y):
            logging.warning(f"Skipping {filename}: mismatch between X ({X.shape[0]}) and y ({len(y)})")
            continue

        np.save(os.path.join(output_dir, "", f"{base}.X.npy"), X)
        np.save(os.path.join(output_dir, "", f"{base}.y.npy"), np.array(y))
        dataset_ids.append(base)

    with open(os.path.join(output_dir, "dataset_list.txt"), "w") as f:
        for base in dataset_ids:
            f.write(base + "\n")

    logging.info(f"Finished processing {len(dataset_ids)} sequences.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
       print("Usage: python3 06.protchnains.py chainlist.txt [outputdir]")
       exit(-1)

    file_list = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output.lda"

    LSTM_make_data(
        file_list=file_list,
        path_to_Datasets_store='',
        path_to_data_store='Databases/AA3_AA1_SS/',
        path_to_frequency_store="Databases/FrequencyExtrapolation/",
        path_to_aaindex_file="Databases/AAindex/aaindex.txt",
        path_to_aaindex_tri_letter_file="Databases/AAindex/aaindex_mutant3.txt",
        dssp_to_index={'H':0,'I':1,'G':2,'E':3,'B':4,'T':5,'S':6,'.':7,'P':7,'*':-100},
        task_set_file='lda.task',
        output_dir=output_dir
    )
