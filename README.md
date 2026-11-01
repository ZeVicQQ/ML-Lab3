# Лабораторная работа №3: Линейные модели для классификации

Скрипт демонстрирует бинарную и многоклассовую линейную классификацию
методами «один против остальных» (One-vs-Rest).

## Требования

- Python 3.10+
- Windows для `start.bat`

## Библиотеки

Указаны в `requirements.txt`:

- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

## Запуск

### Автоматически

1. Дважды кликните `start.bat`.
2. Скрипт создаст `.venv`, установит зависимости и запустит `lab3.py`.

### Вручную

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python lab3.py