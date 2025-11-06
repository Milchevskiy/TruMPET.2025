#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["SKLEARNEX_DISABLE_INFO_MESSAGES"] = "1"
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from sklearnex import patch_sklearn
import logging
logger = logging.getLogger('sklearnex')
logger.setLevel(logging.WARNING)
patch_sklearn()

# ===== CUDA/математика: автоподбор ядер, TF32 =====
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')  # на Ampere+ ускоряет матмул
    except Exception:
        pass

BEST_CUDA = '512_2_cuda.pt'
BEST_CPU  = '512_2_cpu.pt'
DROPOUT   = 0.5
LR        = 1e-4  # learning rate hyperparameter
WDECAY    = 1e-5  # weight decay hyperparameter
train_ids = "train.txt"
val_ids   = "validation.txt"
data_dir  = "output.LSTM"

# ===================== Dataset =====================
class ProteinDataset(Dataset):
    def __init__(self, id_list_path, npy_dir):
        with open(id_list_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        path_x = os.path.join(self.npy_dir, f"{pid}.X.npy")
        path_y = os.path.join(self.npy_dir, f"{pid}.y.npy")

        x = np.load(path_x, mmap_mode='r')
        y = np.load(path_y, mmap_mode='r')

        x = x.astype(np.float32, copy=False)
        if (not x.flags.writeable) or (not x.flags.c_contiguous):
            x = np.array(x, dtype=np.float32, copy=True, order='C')

        if (not y.flags.writeable) or (not y.flags.c_contiguous):
            y = np.array(y, dtype=np.int64, copy=True, order='C')
        else:
            y = y.astype(np.int64, copy=False)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        return x_t, y_t


def collate_batch(batch):
    x_list, y_list = zip(*batch)
    lengths = torch.tensor([len(x) for x in x_list], dtype=torch.int64)
    x_padded = pad_sequence(x_list, batch_first=True)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=-100)
    x_packed = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
    return x_packed, y_padded, lengths


# ===================== Model =====================
class ProteinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=9, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=DROPOUT
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x_packed):
        packed_out, _ = self.lstm(x_packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(output)


# ===================== Eval =====================
@torch.no_grad()
def evaluate(model, dataloader, device, amp_dtype=None, limit_batches: int = 0):
    """
    limit_batches > 0 — оценить только на первых N батчах (ускорение для train-метрик).
    """
    model.eval()
    all_preds, all_labels = [], []
    use_amp = device.type == 'cuda' and amp_dtype is not None
    ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else torch.autocast("cpu", enabled=False)

    seen = 0
    with ctx:
        for x_packed, y_padded, lengths in dataloader:
            x_packed = x_packed.to(device, non_blocking=True)
            y_padded = y_padded.to(device, non_blocking=True)
            output = model(x_packed)

            for i, L in enumerate(lengths):
                true = y_padded[i][:L]
                pred = output[i][:L].argmax(dim=1)
                mask = true != -100
                all_preds.extend(pred[mask].cpu().tolist())
                all_labels.extend(true[mask].cpu().tolist())

            seen += 1
            if limit_batches > 0 and seen >= limit_batches:
                break

    acc = accuracy_score(all_labels, all_preds) if all_labels else float('nan')
    f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else float('nan')
    return acc, f1, all_labels, all_preds


# ===================== Train =====================
def train(model,
          train_loader,
          val_loader,
          device,
          criterion,
          epochs=10,
          lr=5e-5,
          early_stopping_patience=13,
          eval_every=1,
          train_metric_every=3,
          train_metric_subset_batches: int = 0  # 0 = весь train; >0 = ускоренная оценка на N батчах
          ):
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WDECAY, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WDECAY)

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    patience = 0

    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler("train.log")
        fh.setLevel(logging.INFO)
        log.addHandler(fh)

    history = {'loss': [], 'val_acc': [], 'val_f1': [], 'train_acc': [], 'train_f1': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for x_packed, y_padded, _ in pbar:
            x_packed = x_packed.to(device, non_blocking=True)
            y_padded = y_padded.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                output = model(x_packed)
                loss = criterion(output.reshape(-1, output.shape[-1]),
                                 y_padded.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # === Validation по расписанию eval_every ===
        val_acc = val_f1 = float('nan')
        if (epoch + 1) % eval_every == 0:
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            # Лучшие модели и early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if device.type == 'cuda':
                    torch.save(model.state_dict(), BEST_CUDA)
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(cpu_state, BEST_CPU)
                log.info("✅ Лучшая модель сохранена (CUDA и CPU).")
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    log.info("⏹️ Early stopping")
                    break
        else:
            # чтобы длины history были согласованны для графиков
            history['val_acc'].append(np.nan)
            history['val_f1'].append(np.nan)

        # === Train-метрики по расписанию train_metric_every (НЕ зависят от eval_every) ===
        train_acc_ep = train_f1_ep = float('nan')
        if (epoch + 1) % train_metric_every == 0:
            train_acc_ep, train_f1_ep, _, _ = evaluate(
                model, train_loader, device, amp_dtype=amp_dtype,
                limit_batches=train_metric_subset_batches
            )

        # Логирование/история
        history['loss'].append(total_loss)
        history['train_acc'].append(train_acc_ep)
        history['train_f1'].append(train_f1_ep)

        msg_val = f", Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}" if not np.isnan(val_acc) else ""
        msg_train = f" | Train Acc={train_acc_ep:.4f}, Train F1={train_f1_ep:.4f}" if not np.isnan(train_acc_ep) else ""
        log_line = f"Epoch {epoch+1}: Train Loss={total_loss:.4f}{msg_val}{msg_train}  Model={BEST_CUDA}"
        print(log_line)
        log.info(log_line)

    # ===== Графики динамики =====
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.plot(history['loss']);        plt.title("Train Loss");             plt.xlabel("Epoch"); plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2); plt.plot(history['val_acc']);     plt.title("Validation Accuracy");    plt.xlabel("Epoch"); plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3); plt.plot(history['val_f1']);      plt.title("Validation F1");          plt.xlabel("Epoch"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("training_metrics.png")

    # ===== Матрица ошибок по валидации =====
    _, _, y_true, y_pred = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
    cm = confusion_matrix(y_true, y_pred)
    with open('val_confusion_matrix.txt', 'w') as f:
        f.write("Confusion Matrix (validation):\n")
        for row in cm:
            f.write(' '.join(map(str, row)) + '\n')


# ===================== Main =====================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация модели по размеру признаков первой цепи train
    first_id = next(line.strip() for line in open(train_ids) if line.strip())
    sample_x = np.load(os.path.join(data_dir, first_id + ".X.npy"), mmap_mode='r')
    input_size = int(sample_x.shape[1])
    model = ProteinLSTM(input_size=input_size).to(device)

    # Загрузка чекпоинта при наличии
    model_path = BEST_CUDA if device.type == 'cuda' else BEST_CPU
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"🔄 Загрузка сохранённой модели: {model_path}")

    # DataLoaders
    num_workers = min(4, max(1, os.cpu_count() // 4))
    train_ds = ProteinDataset(train_ids, data_dir)
    val_ds   = ProteinDataset(val_ids,  data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_batch,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
        persistent_workers=False, multiprocessing_context="spawn"
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, collate_fn=collate_batch,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2,
        persistent_workers=False, multiprocessing_context="spawn"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Запуск обучения
    # ВАЖНО: train_metric_every=3 — как ты и хотела.
    # train_metric_subset_batches=2, например, чтобы train-метрики считались быстро (на 2 батчах).
    train(model, train_loader, val_loader, device, criterion,
          epochs=900,
          lr=LR,
          early_stopping_patience=13,
          eval_every=1,
          train_metric_every=3,
          train_metric_subset_batches=0  # 0 = считать на всём train
          )
