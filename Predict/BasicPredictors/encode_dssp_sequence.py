import numpy as np


def encode_dssp_sequence(dssp,dssp_to_index):
    encoding = np.zeros((len(dssp)), dtype=np.compat.long)
    for i, aa in enumerate(dssp):
        encoding[i] = int ( dssp_to_index[aa])
    return encoding
