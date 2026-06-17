# BTC Price Forecasting — Walk-Forward Pipeline

Исследовательский конвейер прогнозирования цены биткоина по гетерогенным
финансовым индикаторам. Честно (без утечек «из будущего») сравнивает три семейства
моделей на трёх частотах и оценивает их не только статистически, но и экономически.

- **Таргет:** одношаговая лог-доходность `r_t = ln(P_t) − ln(P_{t−1})` (не уровень цены).
- **Частоты:** дневная, часовая, 5-минутная.
- **Модели:** 4 бустинга (HistGB, LightGBM, XGBoost, CatBoost), 4 нейросетевые
  регрессии (LSTM, GRU, StackedLSTM, CNN-LSTM) и 4 нейросетевых **классификатора
  знака** (взвешенная BCE), плюс наивные бэйзлайны.
- **Валидация:** walk-forward (expanding для дневных, rolling для внутридневных,
  по 5 фолдов), с переподбором гиперпараметров Optuna на каждом фолде; отбор
  признаков и препроцессинг — строго внутри тренировочного фолда.
- **Метрики:** ведущая — Directional Accuracy (+ AUC для классификаторов), MASE,
  плюс SMAPE/MAE на восстановленной цене. Итоговый критерий ценности —
  экономический бэктест против Buy & Hold (комиссии + проскальзывание).

## Структура

```
ml_models.py          # CryptoNet, CryptoNetRegressor, CryptoNetClassifier, метрики, Optuna
walk_forward.py       # движок walk-forward: сплиты, отбор признаков, run_walk_forward
backtest.py           # экономическая симуляция (PnL, Sharpe, Max Drawdown vs Buy&Hold)
get_data.py           # сбор данных (DataMaker): Yahoo Finance, FRED, Fear&Greed, BTC supply
mmda_btc_report.ipynb # основной отчёт/пайплайн (запускать сверху вниз)
new_day_df.csv        # подготовленные данные — дневные
new_hour_df.csv       # подготовленные данные — часовые
new_5min_df.csv       # подготовленные данные — 5-минутные
requirements.txt      # зависимости
```

## Установка

Требуется Python 3.11. `numpy` закреплён на `<2.0` (требование `catboost`).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU (опционально, но желательно для нейросетей): убедитесь, что PyTorch видит CUDA —
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Если выводит `False` — переустановите torch под вашу CUDA, напр.
`pip install torch --index-url https://download.pytorch.org/whl/cu121`.
Бустинги считаются на CPU (это оптимально при таком объёме данных), нейросети — на GPU.

## Запуск

```bash
jupyter notebook mmda_btc_report.ipynb
```
Выполнить ячейки сверху вниз. CSV-файлы уже включены, поэтому скачивание данных не
требуется. Полный прогон ресурсоёмкий (walk-forward × Optuna × 12 моделей × 3 частоты);
на GPU — порядка десятков минут.

### Ключевые «ручки» (ячейка конфигурации)
- `SPLIT_CONFIG` — схема и число фолдов walk-forward по частотам.
- `WF_OPTUNA_TRIALS` / `WF_NN_OPTUNA_TRIALS` — число trials Optuna на фолд (бустинги / нейросети).
- `RUN_NN_CLF` — включить/выключить дорожку классификаторов знака.
- `TOP_FEATURES` — пул признаков-кандидатов (пофолдово отбирается top-`WF_TOP_K`).
- `NN_CONFIG` — длина окна / эпохи / batch для нейросетей.

## Данные

CSV получены скриптом `get_data.py` (класс `DataMaker`): Yahoo Finance (крипта,
индексы, сырьё, FX, акции), FRED (макро), индекс страха и жадности, BTC supply,
технические индикаторы (RSI/MACD/ATR/OBV). Для пересборки нужен ключ FRED:

```bash
export FRED_API_KEY=<your_key>     # https://fred.stlouisfed.org/docs/api/api_key.html
```

Все экзогенные ряды лагируются на 1 бар уже на этапе сбора (защита от look-ahead).

## Воспроизводимость / Git LFS

`new_hour_df.csv` (~25 МБ) и `new_5min_df.csv` (~20 МБ) большие. Рекомендуется
хранить их через [Git LFS](https://git-lfs.com):

```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
```
(выполнить до первого `git add` самих CSV). Альтернатива — выложить данные в Release
и регенерировать через `get_data.py`.
