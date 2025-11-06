#!/usr/bin/env python3
import sys
import os
import logging
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'BasicPredictors/'))

TASKSET = 'postLDA.task'

from _CommonConstants import _CommonConstants
from _create_predictors_set_common import _create_predictors_set_common
from create_predictors_MLP import create_predictors_MLP
from encode_dssp_sequence import encode_dssp_sequence
from iter_RawData import iter_RawData

import esm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_esm2_embeddings(aa_sequence, esm_model, batch_converter, device):
    data = [("protein", aa_sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    return token_representations[0, 1:len(aa_sequence)+1].cpu().numpy()


def LSTM_make_data(
    file_list,
    path_to_Datasets_store,
    path_to_data_store,
    path_to_frequency_store,
    path_to_aaindex_file,
    path_to_aaindex_tri_letter_file,
    dssp_to_index,
    task_set_file,
    output_dir="output"
):
    os.makedirs(os.path.join(output_dir, "X"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "y"), exist_ok=True)

    logging.basicConfig(filename=os.path.join(output_dir, "data_generation.log"),
                        filemode='w',
                        level=logging.INFO)

#    torch.hub.set_dir("./Models/ESM2")
    device = get_device()
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

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

        try:
            X_phys = create_predictors_MLP(objects, aa1, aa3, common)
            X_esm = get_esm2_embeddings(aa1, esm_model, batch_converter, device)
        except Exception as e:
            logging.warning(f"⚠️ Skipping {filename}: ESM2 embedding failed: {e}")
            continue

        if X_phys.shape[0] != X_esm.shape[0]:
            logging.warning(f"⚠️ Skipping {filename}: shape mismatch {X_phys.shape[0]} vs {X_esm.shape[0]}")
            continue

        X = np.concatenate([X_phys, X_esm], axis=1)
        y = encode_dssp_sequence(dssp, dssp_to_index)

        if X.shape[0] != len(y):
            logging.warning(f"⚠️ Skipping {filename}: mismatch between X ({X.shape[0]}) and y ({len(y)})")
            continue

        np.save(os.path.join(output_dir, "", f"{base}.X.npy"), X)
        np.save(os.path.join(output_dir, "", f"{base}.y.npy"), np.array(y))
        dataset_ids.append(base)

    with open(os.path.join(output_dir, "dataset_list.txt"), "w") as f:
        for base in dataset_ids:
            f.write(base + "\n")

    logging.info(f"✅ Finished processing {len(dataset_ids)} sequences.")


if __name__ == "__main__":
    import sys
    file_list = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    LSTM_make_data(
        file_list=file_list,
        path_to_Datasets_store='',
        path_to_data_store='/home/milch/_TMP/_____/Final_SS/DATA/aa3_aa1_ss/',
        path_to_frequency_store="/home/milch/_TMP/_____/Final_SS/DATA/FrequencyExtrapolation/",
        path_to_aaindex_file="/home/milch/_TMP/_____/Final_SS/DATA/aaindex/aaindex.data/aaindex.txt",
        path_to_aaindex_tri_letter_file="/home/milch/_TMP/_____/Final_SS/DATA/aaindex/aaindex.data/aaindex_mutant3.txt",
        dssp_to_index={'H':0,'I':1,'G':2,'E':3,'B':4,'T':5,'S':6,'.':7,'P':7,'*':-100},
        task_set_file=TASKSET,
        output_dir=output_dir
    )