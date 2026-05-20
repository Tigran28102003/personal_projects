# Прогнозирование ВРП субъектов РФ с помощью ансамблевых методов МО

Магистерская диссертация, НИУ ВШЭ, ФКН, ОП «Анализ больших данных», 2026.

**Тема**: Forecasting Gross Regional Product of Russian Federation Subjects Using Ensemble Machine Learning Methods  
**Автор**: Семенюченко Артём Геннадьевич  
**Научный руководитель**: Паточенко Евгений Анатольевич

---

## Результаты

Кросс-валидация на расширяющемся окне (9 fold'ов, тестовые годы 2015–2023, 85 регионов):

| Модель | RMSE | MAE | OOS R² |
|--------|------|-----|--------|
| **CatBoost** | **0.0395** | **0.0300** | **0.053** |
| LightGBM | 0.0410 | 0.0314 | −0.005 |
| Наивный прогноз | 0.0419 | 0.0331 | 0.000 |
| AR(1) | 0.0487 | 0.0336 | −0.40 |
| Ridge с FE | 0.0699 | 0.0603 | −2.51 |
| GPBoost | 0.1937 | 0.1399 | −34.91 |

CatBoost снижает RMSE относительно наивного прогноза на **5.8%**. Топ-5 признаков по SHAP: реальная ключевая ставка ЦБ, диапазон и волатильность курса USD/RUB, индикатор пандемии 2020 г., изменение ключевой ставки.

---

## Структура репозитория

```
disser_epta/
├── grp_forecast/               # основной пакет
│   ├── main.ipynb              # все эксперименты (CV, ablation, SHAP, DM-тест)
│   ├── features.py             # загрузка данных и feature engineering
│   ├── models.py               # реализации моделей (Naive, AR1, Ridge, LGBM, CatBoost, GPBoost, EmbMLP)
│   ├── validation.py           # кросс-валидация и метрики
│   ├── tests.py                # проверки на leakage, ablation, robustness checks
│   ├── visualization.py        # построение графиков
│   ├── config.yaml             # гиперпараметры и пути
│   ├── figures/                # PNG-графики (метрики, SHAP, ошибки, robustness)
│   ├── data/
│   │   ├── raw/                # исходные CSV/XLSX по регионам
│   │   └── processed/          # parquet-предсказания по fold'ам
│   └── make_data/              # сборка панельного датасета
│
├── texts_&_presentation/       # финальные PDF-версии
│   ├── dissert_revised.pdf
│   ├── predefense_presentation.pdf
│   └── predefense_speech.md
│
├── dissert_revised.tex         # исходник диссертации (XeLaTeX)
├── predefense_presentation.tex # исходник презентации (Beamer)
├── pipeline_review_report.md   # аналитический отчёт по пайплайну
└── readme.md
```

---

## Запуск

**Зависимости** (Python 3.10+):
```bash
pip install -r grp_forecast/requirements.txt
```

**Все эксперименты** — последовательный запуск ячеек `grp_forecast/main.ipynb`.  
Разделы ноутбука:
1. Загрузка и описание данных
2. Feature engineering и построение fold'ов
3. Кросс-валидация 6 моделей (с Optuna-тюнингом)
4. Ablation study (сценарии A0–A3)
5. Robustness checks (RC1: без выбросов, RC2: кризис vs. норма)
6. SHAP-интерпретация (global, group, local)
7. Анализ ошибок (по регионам и годам)
8. Итоговые выводы
9. EmbMLP (нейросеть с эмбеддингами регионов)
10. GPBoost с включённой GP-компонентой
11. Ансамбль CatBoost + LightGBM
12. Тест Дибольда–Мариано

**Компиляция диссертации** (требует XeLaTeX и шрифты DejaVu):
```bash
xelatex dissert_revised.tex
xelatex dissert_revised.tex  # второй проход для TOC
```
