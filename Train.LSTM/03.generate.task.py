#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Читает файл настроек (<name_tune>) и генерирует строки вида:
_c_FourierSmoothed_tail ANDN920101 1.2 3 1
(без кавычек всегда)

Результат пишется в <name_tune>.txt (если явно не задан outfile).
props-file: в каждой строке сначала код свойства, затем описание (игнорируется).
Комментарии после '#' отрезаются.
"""

from __future__ import annotations
import sys
import os
from typing import Iterable, List

# ===================== Утилиты =====================

def frange_inclusive(start: float, stop: float, step: float) -> Iterable[float]:
    """Перебор вещественных значений по сетке с включением правой границы."""
    if step <= 0:
        raise ValueError("period step must be > 0")
    x = start
    while x <= stop + 1e-12:
        yield float(f"{x:.12g}")  # сглаживаем накопление ошибки
        x += step
    # если не попали точно в stop — гарантируем включение
    if stop > start and abs((stop - start) % step) > 1e-9:
        yield float(f"{stop:.12g}")

def irange_inclusive(start: int, stop: int, step: int) -> Iterable[int]:
    """Перебор целых значений по сетке с включением правой границы."""
    if step <= 0:
        raise ValueError("integer step must be > 0")
    if start > stop:
        return
    v = start
    while v <= stop:
        yield v
        v += step

def read_props(path: str, strip_comments: bool) -> List[str]:
    """
    Читает список свойств. Формат строки:
      <CODE> <optional description...>
    Берётся только первый токен <CODE>. Комментарии после '#' отрезаются.
    """
    props: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if strip_comments and "#" in s:
                s = s.split("#", 1)[0].strip()
                if not s:
                    continue
            cleaned = s.replace(",", " ").replace(";", " ").strip()
            parts = cleaned.split()
            if not parts:
                continue
            props.append(parts[0])
    return props

def fmt_float(x: float) -> str:
    """Компактный вывод float без лишних нулей."""
    return f"{x:g}"

# ===================== Парсер tune-файла =====================

def parse_tune(path: str) -> dict:
    """
    Поддерживаемые ключи (без дефисов, регистр неважен):
      type              <string>                      (по умолчанию _c_FourierSmoothed_tail)
      props-file        <path>
      period            <min> <max> <step>            (float)
      nperiods          <min> <max> <step>            (int)
      degree            <min> <max> <step>            (int)
      strip-comments    true|false                    (default true)
      outfile           <path>                        (по умолчанию <name_tune>.txt)

    Принимаются варианты с '-' или '--' (например, "-props-file").
    Комментарии: строки, начинающиеся с #, и хвост после #.
    Формат: <key> <values...> (через пробелы).
    """
    cfg = {
        "type": "_c_FourierSmoothed_tail",
        "strip-comments": True,
        "outfile": None,
    }

    def norm_key(k: str) -> str:
        k = k.strip()
        if k.startswith("--"):
            k = k[2:]
        elif k.startswith("-"):
            k = k[1:]
        return k.lower()

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
            parts = line.split()
            if not parts:
                continue
            key = norm_key(parts[0])
            vals = parts[1:]

            if key in ("type", "props-file", "outfile"):
                if not vals:
                    raise ValueError(f"Key '{key}' requires a value")
                cfg[key] = vals[0]
            elif key in ("period", "nperiods", "degree"):
                if len(vals) != 3:
                    raise ValueError(f"Key '{key}' must have 3 values: <min> <max> <step>")
                if key == "period":
                    cfg["period"] = (float(vals[0]), float(vals[1]), float(vals[2]))
                else:
                    cfg[key] = (int(vals[0]), int(vals[1]), int(vals[2]))
            elif key == "strip-comments":
                if not vals:
                    raise ValueError(f"Key '{key}' requires true|false")
                v = vals[0].lower()
                if v in ("true", "1", "yes", "on"):
                    cfg[key] = True
                elif v in ("false", "0", "no", "off"):
                    cfg[key] = False
                else:
                    raise ValueError(f"Key '{key}' expects true/false, got '{vals[0]}'")
            else:
                raise ValueError(f"Unknown key '{key}' in {path}: '{raw.strip()}'")

    # обязательные поля
    must = ["props-file", "period", "nperiods", "degree"]
    missing = [m for m in must if m not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in tune file: {', '.join(missing)}")

    return cfg

# ===================== Основная логика =====================

def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {os.path.basename(argv[0])} <name_tune>", file=sys.stderr)
        return 2
    tune_path = argv[1]
    if not os.path.isfile(tune_path):
        print(f"Tune file not found: {tune_path}", file=sys.stderr)
        return 2

    cfg = parse_tune(tune_path)
    props = read_props(cfg["props-file"], strip_comments=cfg["strip-comments"])
    if not props:
        print("No properties found in props-file (after stripping/ignoring).", file=sys.stderr)
        return 1

    per_min, per_max, per_step = cfg["period"]
    np_min, np_max, np_step = cfg["nperiods"]
    dg_min, dg_max, dg_step = cfg["degree"]

    out_path = cfg["outfile"] or (tune_path + ".txt")

    with open(out_path, "w", encoding="utf-8") as out:
        for prop in props:
            for per in frange_inclusive(per_min, per_max, per_step):
                for nper in irange_inclusive(np_min, np_max, np_step):
                    for deg in irange_inclusive(dg_min, dg_max, dg_step):
                        # ВСЕГДА без кавычек:
                        core = f"{cfg['type']} {prop} {fmt_float(per)} {nper} {deg}"
                        out.write(core + "\n")

    print(f"Done: {out_path}  (props={len(props)})")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
