#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import logging

# Пути к файлам
DATA_FILE  = "lda.csv"
NAMES_FILE = "lda.task"

# Загрузка данных с метаинформацией
def load_data_with_meta(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    meta = lines[0].split()
    n_features = int(meta[2])
    data_lines = lines[1:]
    data = [list(map(float, line.strip().split())) for line in data_lines]

    df = pd.DataFrame(data, columns=['class'] + [f'x{i+1}' for i in range(n_features)])
    return df

# Загрузка имён признаков
def load_feature_names(names_path):
    with open(names_path, encoding='utf-8') as f:
        lines = f.readlines()
    names = [line.strip() for line in lines]
    return names

# Оценка одного признака с защитой от ошибок
def evaluate_feature(selected, feature, X, y):
    candidate = selected + [feature]
    X_candidate = X[candidate]

    if X_candidate.shape[1] == 0 or np.any(X_candidate.std(ddof=0) == 0):
        logging.warning(f"Признак {feature} исключён: нулевая дисперсия или пустой набор")
        return feature, -1

    try:
        model = LinearDiscriminantAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_candidate, y)
        score = model.score(X_candidate, y)
        return feature, score
    except Exception as e:
        logging.warning(f"Ошибка на признаке {feature}: {e}")
        return feature, -1

# Greedy stepwise LDA
def greedy_stepwise_lda(X, y, feature_names=None, max_features=50,
                        min_improvement=1e-3, max_score=0.90, n_jobs=-1):
    selected = []
    remaining = list(X.columns)
    last_score = 0
    history = []

    logging.info(f"Запуск greedy stepwise LDA на {len(remaining)} признаках...")

    for step in range(max_features):
        print(f"\n🚀 Шаг {step + 1}: перебор {len(remaining)} признаков...")
        try:
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_feature)(selected, feature, X, y) for feature in remaining
            )
        except Exception as e:
            logging.error(f"Ошибка в параллельной обработке: {e}")
            break

        valid_results = [(f, s) for f, s in results if s >= 0]
        if not valid_results:
            logging.warning("Все признаки на этом шаге дали ошибки. Отбор прекращён.")
            break

        best_feature, best_score = max(valid_results, key=lambda x: x[1])
        improvement = best_score - last_score

        if improvement < min_improvement:
            print(f"⏹️ Улучшение < {min_improvement:.2e}. Отбор завершён.")
            break
        if best_score >= max_score:
            print(f"⏹️ Достигнута точность {best_score:.4f} ≥ {max_score}. Отбор завершён.")
            selected.append(best_feature)
            history.append((best_feature, best_score, improvement))
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        history.append((best_feature, best_score, improvement))
        print(f"✅ Добавлен: {best_feature:6s} → точность: {best_score:.4f} (+{improvement:.4f})")
        last_score = best_score

    return selected, history

# Отчёт и сохранение результатов
def report_results(selected, history, feature_names=None,
                   save_path_txt="greedy_selected_features.txt",
                   save_path_csv="greedy_selection_history.csv"):

    scores = [h[1] for h in history]
    steps = list(range(1, len(scores) + 1))
    improvements = [h[2] for h in history]

    # TXT-таблица
    with open(save_path_txt, "w", encoding='utf-8') as f:
        f.write("Признак\tТочность\tПрирост\n")
        for i, (colname, score, gain) in enumerate(history):
            readable = feature_names[int(colname[1:]) - 1] if feature_names else colname
            f.write(f"{readable}\t{score:.4f}\t{gain:.4f}\n")

    # CSV-таблица
    df_out = pd.DataFrame([
        {
            "Step": i + 1,
            "Feature": feature_names[int(colname[1:]) - 1] if feature_names else colname,
            "Score": score,
            "Improvement": gain
        }
        for i, (colname, score, gain) in enumerate(history)
    ])
    df_out.to_csv(save_path_csv, index=False)

    logging.info(f"Сохранён TXT-отчёт: {save_path_txt}")
    logging.info(f"Сохранён CSV-отчёт: {save_path_csv}")

    print(f"\n📁 Отчёт сохранён в {save_path_txt}")
    print(f"📄 CSV сохранён в {save_path_csv}")

# Основной блок
if __name__ == "__main__":
    # Настройка логгирования
    logging.basicConfig(
        filename='greedy_stepwise_lda.log',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("🔍 Запуск greedy_stepwise_lda")


    df = load_data_with_meta(DATA_FILE)
    feature_names = load_feature_names(NAMES_FILE)
    X = df.drop(columns='class')
    y = df['class']

    selected, history = greedy_stepwise_lda(X, y,
                                            feature_names=feature_names,
                                            max_features=100,
                                            min_improvement=5e-9,
                                            max_score=0.75,
                                            n_jobs=-1)

    report_results(selected, history, feature_names)
