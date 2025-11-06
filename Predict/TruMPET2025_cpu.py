#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ========= Default Settings =========
MODEL_PATH      = "Models/mix/1024_4_cpu.pt" # mix model provides 84.46 Q8 prediction accuracy with 583 additional descriptors
TASK_SET_FILE   = "Models/mix/1024_4.task"
#MODEL_PATH      = "Models/LDA/512_2_cpu.pt" # LDA model provides 83.11 Q8 prediction accuracy with 153 additional descriptors
#TASK_SET_FILE   = "Models/LDA/512_2.task"
OUT_DIR = "results"
LOG_LEVEL = "INFO"
# Databases Paths
PATH_TO_FREQUENCY_STORE    = "Databases/FrequencyExtrapolation/"
PATH_TO_AAINDEX_FILE       = "Databases/AAindex/aaindex.txt"
PATH_TO_AAINDEX_TRI_LETTER = "Databases/AAindex/aaindex_mutant3.txt"
# ==========================================

# === Descriptors type code ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'BasicPredictors/'))
from _CommonConstants import _CommonConstants
from _create_predictors_set_common import _create_predictors_set_common
from create_predictors_MLP import create_predictors_MLP

# ESM2
try:
    import esm
except Exception:
    esm = None

# --- DSSP Q8 dictionaries ---
index_to_dssp = {v:k for v,k in enumerate("HIGEBTS-")}  # 0..7 → symbol

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# FIX: reverse dict for aa3 from aa1 encoding (FASTA)
AA1_TO_AA3 = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS",
    "Q":"GLN","E":"GLU","G":"GLY","H":"HIS","I":"ILE",
    "L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO",
    "S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
    # noncanonic/unknown symbols:
    "U":"UNK","O":"UNK","B":"UNK","Z":"UNK","J":"UNK","X":"UNK","*":"UNK","-":"UNK"
}

# --- Модель LSTM ---
class ProteinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=9, num_layers=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x_packed):
        packed_out, _ = self.lstm(x_packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out_padded)

# --- чтение .data ---
def read_aa1_aa3_any_order(filepath):
    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{filepath}: нужно минимум 2 строки (aa1 и aa3).")

    l0, l1 = lines[0], lines[1]
    def looks_like_aa3(s):
        toks = s.split()
        return len(toks) >= 2 and all(len(t)==3 for t in toks)

    if looks_like_aa3(l0) and not looks_like_aa3(l1):
        aa3_list = l0.split(); aa1 = ''.join(l1.split())
    elif looks_like_aa3(l1) and not looks_like_aa3(l0):
        aa3_list = l1.split(); aa1 = ''.join(l0.split())
    else:
        if not looks_like_aa3(l0) and not looks_like_aa3(l1):
            aa1 = ''.join(l0.split()); aa3_list = l1.split() if looks_like_aa3(l1) else []
        else:
            aa3_list = l0.split(); aa1 = ''.join(l1.split())

    if not aa1 and aa3_list:
        aa1 = ''.join(AA3_TO_AA1.get(t.upper(), 'X') for t in aa3_list)
    if not aa1:
        raise ValueError(f"{filepath}: не удалось распознать однобуквенную последовательность.")

    # FIX: если aa3_list отсутствует/короче — дособрать из aa1
    if not aa3_list or len(aa3_list) != len(aa1):
        aa3_list = [AA1_TO_AA3.get(ch, "UNK") for ch in aa1]

    return aa3_list, aa1

# --- чтение FASTA ---
def read_fasta_aa1(filepath):
    aa = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            aa.append(line)
    aa1 = ''.join(aa).replace(" ", "").upper()
    aa1 = ''.join(ch for ch in aa1 if ch.isalpha() or ch in "*-").upper()
    if not aa1:
        raise ValueError(f"{filepath}: не удалось прочитать последовательность из FASTA.")
    return aa1

def aa1_to_aa3_list(aa1):
    return [AA1_TO_AA3.get(ch, "UNK") for ch in aa1]

def get_device():
    return ("cpu")

def get_esm2_model_and_converter():
    if esm is None:
        raise RuntimeError("fair-esm не установлен")
# set directory with ESM2 model. If it is not pre-downloaded, it will be downloaded automagically
    torch.hub.set_dir("./Models/ESM2")
    device_esm = get_device()
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device_esm)
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter, device_esm

def get_esm2_embeddings(aa_seq, esm_model, batch_converter, device):
    data = [("protein", aa_seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    tok = results["representations"][33]
    return tok[0, 1:len(aa_seq)+1].detach().cpu().numpy()

def run_inference_single(aa1, aa3_list, out_basename, lstm_model, esm_pack):
    # 1) фичи (physicochemical)
    common = _CommonConstants(
        PATH_TO_FREQUENCY_STORE,
        PATH_TO_AAINDEX_FILE,
        PATH_TO_AAINDEX_TRI_LETTER,
        log_level=LOG_LEVEL
    )
    objects = _create_predictors_set_common(TASK_SET_FILE, common)

    # FIX: ensure the equal lengths (important for the Three_Letter_Mode)
    if not aa3_list or len(aa3_list) != len(aa1):
        aa3_list = aa1_to_aa3_list(aa1)

    X_phys = create_predictors_MLP(objects, aa1, aa3_list, common)   # (L, D_phys)

    # 2) ESM2
    esm_model, batch_converter, device_esm = esm_pack
    X_esm = get_esm2_embeddings(aa1, esm_model, batch_converter, device_esm)  # (L, 1280)
    if X_esm.shape[0] != X_phys.shape[0]:
        raise RuntimeError(f"Length mismatch: phys={X_phys.shape[0]} vs esm={X_esm.shape[0]}")
    X = np.concatenate([X_phys, X_esm], axis=1)

    # 3) inference LSTM
    L, F = X.shape
    device = next(lstm_model.parameters()).device
    x_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # (1, L, F)
    lengths = torch.tensor([L], dtype=torch.long, device=device)
    x_packed = pack_padded_sequence(x_t, lengths, batch_first=True, enforce_sorted=False)
    with torch.no_grad():
        logits = lstm_model(x_packed)
    pred_idx = logits[0].argmax(dim=1).detach().cpu().numpy()
    pred_ss = ''.join(index_to_dssp.get(int(i), '*') for i in pred_idx)

    # 4) output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{out_basename}.pred.txt")
    with open(out_path, "w") as fw:
        fw.write(aa1 + "\n")
        fw.write(pred_ss + "\n")

    print(f" {out_basename}: L={L}, D={F} → {out_path}")

def expand_paths(paths_or_patterns):
    seen = set()
    result = []
    for p in paths_or_patterns or []:
        matches = glob.glob(p) or [p]
        for m in matches:
            m = os.path.abspath(m)
            if os.path.isfile(m) and m not in seen:
                seen.add(m)
                result.append(m)
    return result

def parse_args():
    p = argparse.ArgumentParser(description="Пакетный инференс DSSP-8 по .data и/или FASTA.")
    p.add_argument("-d", "--data",  nargs="+", metavar="FILE", help=".data файлы (aa1/aa3)")
    p.add_argument("-f", "--fasta", nargs="+", metavar="FILE", help="FASTA файлы")
    p.add_argument("--model",  default=MODEL_PATH, help="Путь к .pt модели LSTM")
    p.add_argument("--outdir", default=OUT_DIR,   help="Директория для результатов")
    args = p.parse_args()
    if not args.data and not args.fasta:
        p.error("At least one input file should be set with: -d and/or -f")
    return args

def main():
    global MODEL_PATH, OUT_DIR
    args = parse_args()
    MODEL_PATH = args.model
    OUT_DIR = args.outdir

    data_files  = expand_paths(args.data)
    fasta_files = expand_paths(args.fasta)
    all_inputs = []
    for fp in data_files:
        all_inputs.append(("data", fp))
    for fp in fasta_files:
        all_inputs.append(("fasta", fp))

    # Подготовка: ESM и LSTM грузим один раз
    esm_pack = get_esm2_model_and_converter()

    # FIX: корректное получение пробного кейса (без двойных вызовов)
    first_kind, first_fp = all_inputs[0]
    if first_kind == "data":
        aa3_list_first, aa1_first = read_aa1_aa3_any_order(first_fp)
    else:
        aa1_first = read_fasta_aa1(first_fp)
        aa3_list_first = aa1_to_aa3_list(aa1_first)

    # Получим размерность признаков на первом кейсе
    common_probe = _CommonConstants(
        PATH_TO_FREQUENCY_STORE,
        PATH_TO_AAINDEX_FILE,
        PATH_TO_AAINDEX_TRI_LETTER,
        log_level=LOG_LEVEL
    )
    objects_probe = _create_predictors_set_common(TASK_SET_FILE, common_probe)
    X_phys_probe = create_predictors_MLP(objects_probe, aa1_first, aa3_list_first, common_probe)
    X_esm_probe = get_esm2_embeddings(aa1_first, *esm_pack)
    F_total = X_phys_probe.shape[1] + X_esm_probe.shape[1]

    # Готовим LSTM
    device = "cpu"
    lstm_model = ProteinLSTM(input_size=F_total).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    lstm_model.load_state_dict(state)
    lstm_model.eval()

    # Основной цикл
    for kind, fp in all_inputs:
        base_id = os.path.splitext(os.path.basename(fp))[0]
        try:
            if kind == "data":
                aa3_list, aa1 = read_aa1_aa3_any_order(fp)
            else:
                aa1 = read_fasta_aa1(fp)
                aa3_list = aa1_to_aa3_list(aa1)  # FIX: генерируем aa3 и здесь
            run_inference_single(aa1, aa3_list, base_id, lstm_model, esm_pack)
        except Exception as e:
            print(f"❌ Error for {base_id}: {e}")

if __name__ == "__main__":
    main()
