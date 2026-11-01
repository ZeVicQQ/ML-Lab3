import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "lab3_results"

if RESULT_DIR.exists():
    shutil.rmtree(RESULT_DIR)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Генерация случайной задачи классификации
X, y = make_classification(
    n_samples=600,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.5,
    flip_y=0.05,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Модели и сетка параметра C
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

model_factories = {
    "LogisticRegression": lambda C: OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    ),
    "LinearSVC": lambda C: OneVsRestClassifier(
        LinearSVC(C=C, max_iter=20000, dual="auto", random_state=42)
    ),
}

results = []
best_models = {}
best_C = {}

for model_name, factory in model_factories.items():
    for C in C_values:
        clf = factory(C)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results.append({"model": model_name, "C": C, "accuracy": acc})
        print(f"{model_name}, C={C}: accuracy={acc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_DIR / "c_experiment.csv", index=False, encoding="utf-8-sig")

# 3. Выбор оптимального C для каждой модели
for model_name, factory in model_factories.items():
    sub = results_df[results_df["model"] == model_name]
    best_row = sub.loc[sub["accuracy"].idxmax()]
    best_C[model_name] = best_row["C"]
    best_models[model_name] = factory(best_row["C"])
    best_models[model_name].fit(X_train, y_train)
    print(
        f"Лучший C для {model_name}: {best_row['C']} "
        f"(accuracy={best_row['accuracy']:.4f})"
    )

# 4. Построение графиков
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Лабораторная работа №3: линейная классификация",
    fontsize=16,
    fontweight="bold",
)

# 4.1. Влияние C на accuracy
ax1 = plt.subplot(2, 2, 1)
for model_name in model_factories:
    sub = results_df[results_df["model"] == model_name]
    ax1.plot(sub["C"], sub["accuracy"], marker="o", label=model_name)

ax1.set_xscale("log")
ax1.set_xlabel("C (логарифмическая шкала)")
ax1.set_ylabel("Accuracy на тесте")
ax1.set_title("Влияние параметра C")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# 4.2. Сравнение коэффициентов для класса 0
ax2 = plt.subplot(2, 2, 2)
for model_name, marker in [("LogisticRegression", "o"), ("LinearSVC", "s")]:
    clf = best_models[model_name]
    # OneVsRestClassifier хранит бинарные модели в estimators_
    coef = clf.estimators_[0].coef_[0]
    ax2.plot(coef, marker=marker, markersize=5, label=f"{model_name}, класс 0")

ax2.set_xlabel("Индекс признака")
ax2.set_ylabel("Значение коэффициента")
ax2.set_title("Сравнение коэффициентов (класс 0)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

# 4.3–4.4. Кривые обучения
train_sizes = np.linspace(0.1, 1.0, 10)

for idx, (model_name, clf) in enumerate(best_models.items(), start=3):
    ax = plt.subplot(2, 2, idx)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        clf,
        X,
        y,
        train_sizes=train_sizes,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    ax.plot(train_sizes_abs, train_mean, "o-", label="Обучение")
    ax.plot(train_sizes_abs, test_mean, "s-", label="Кросс-валидация")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Кривая обучения: {model_name} (C={best_C[model_name]})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(RESULT_DIR / "combined_graphs.png", dpi=150)
plt.show()

# 5. Отчёты по классификации
for model_name, clf in best_models.items():
    print(f"\nОтчёт по классификации для {model_name}:")
    print(classification_report(y_test, clf.predict(X_test)))