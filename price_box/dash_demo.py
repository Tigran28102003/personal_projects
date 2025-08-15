import streamlit as st
import numpy as np
import pandas as pd
import random

# ------------------- Генерация справочников -------------------

# Категории техники
categories = [
    "Смартфоны", "Ноутбуки", "Планшеты", "Смарт-часы", "Наушники",
    "Телевизоры", "МФУ/Принтеры", "Мониторы", "Клавиатуры", "Мыши"
]

# 50 наименований товаров с уникальным названием и категорией
product_names = []
product_cats = []
for i in range(50):
    category = random.choice(categories)
    prod_name = f"{category} {chr(65 + i%10)}-{1000 + i}"
    product_names.append(prod_name)
    product_cats.append(category)

# ------------------- Основные параметры и генерация -------------------

n_days = 30
start_date = pd.to_datetime("2024-01-01")
np.random.seed(42)
random.seed(42)

dates = pd.date_range(start=start_date, periods=n_days, freq='D')

# Для простоты — 200-600 продаж, 6-20 продаж в день по товару, разная цена
base_sales = np.random.poisson(lam=12, size=(n_days, 50))
base_returns = np.random.binomial(base_sales, 0.12)
base_revenue = base_sales * np.random.randint(4_000, 60_000, size=(n_days, 50))
base_rating = np.random.beta(a=7, b=2, size=(n_days, 50)) * 4 + 1
base_marketing = np.random.gamma(shape=2, scale=400, size=(n_days, 50))
base_conversion = np.random.beta(a=2.5, b=12, size=(n_days, 50))

# Собираем общую таблицу по дням и товарам
records = []
for day in range(n_days):
    for prod_idx in range(50):
        records.append({
            "Дата": dates[day].strftime('%Y-%m-%d'),
            "Товар": product_names[prod_idx],
            "Категория": product_cats[prod_idx],
            "Продажи": int(base_sales[day, prod_idx]),
            "Выручка": float(base_revenue[day, prod_idx]),
            "Возвраты": int(base_returns[day, prod_idx]),
            "Средний рейтинг": float(base_rating[day, prod_idx]),
            "Расходы на маркетинг": float(base_marketing[day, prod_idx]),
            "Конверсия": float(base_conversion[day, prod_idx]),
        })
df = pd.DataFrame(records)

# ------------------- Сводные метрики по всему магазину -------------------

df["Средний чек"] = np.where(df["Продажи"] > 0, df["Выручка"] / df["Продажи"], 0)
df["Прибыль"] = df["Выручка"] - (df["Расходы на маркетинг"] + df["Возвраты"] * df["Выручка"].mean() * 0.1)

# Для фильтрации
df["Дата_dt"] = pd.to_datetime(df["Дата"])

# Временной фильтр
start_filter, end_filter = st.date_input(
    "Выберите временной период для анализа",
    value=[start_date, start_date + pd.Timedelta(days=n_days-1)],
    min_value=start_date,
    max_value=start_date + pd.Timedelta(days=n_days-1)
)

filt_df = df[(df["Дата_dt"] >= pd.to_datetime(start_filter)) & (df["Дата_dt"] <= pd.to_datetime(end_filter))]

# Сводные значения для карточек
agg = filt_df.groupby("Дата").agg({
    "Продажи": "sum",
    "Выручка": "sum",
    "Прибыль": "sum",
    "Возвраты": "sum",
    "Средний чек": "mean",
    "Средний рейтинг": "mean",
    "Расходы на маркетинг": "sum",
    "Конверсия": "mean"  # Эту строку добавить
}).reset_index()


items_sold = int(agg["Продажи"].sum())
total_revenue = agg["Выручка"].sum()
total_profit = agg["Прибыль"].sum()
total_returns = agg["Возвраты"].sum()
return_rate = total_returns / items_sold * 100 if items_sold > 0 else 0
avg_check = agg["Средний чек"].mean()
avg_rating = agg["Средний рейтинг"].mean()
avg_conv = agg["Конверция"].mean() * 100
total_marketing = agg["Расходы на маркетинг"].sum()

st.title("Дашборд селлера маркетплейса: техника (демо-данные)")

# Карточки с ключевыми метриками
col1, col2, col3, col4 = st.columns(4)
col1.metric("Выручка, ₽", f"{total_revenue:,.0f}")
col2.metric("Чистая прибыль, ₽", f"{total_profit:,.0f}")
col3.metric("Продано товаров", items_sold)
col4.metric("Средний чек, ₽", f"{avg_check:.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Доля возвратов, %", f"{return_rate:.1f}")
col6.metric("Средний рейтинг", f"{avg_rating:.2f}")
col7.metric("Конверсия, %", f"{avg_conv:.2f}")
col8.metric("Расходы на рекламу, ₽", f"{total_marketing:,.0f}")

# ------ Вариативные графики ------

# 1. Динамика выручки, продаж, прибыли по дням (по всем товарам)
st.subheader("Динамика ключевых финансовых показателей по дням")
st.line_chart(
    agg.set_index("Дата")[["Выручка", "Продажи", "Прибыль"]]
)

# 2. ТОП-5 категорий по выручке и продажам
st.subheader("ТОП-5 категорий по выручке и продажам")
top_cats = (
    filt_df.groupby("Категория")
    .agg({"Продажи": "sum", "Выручка": "sum"})
    .sort_values("Выручка", ascending=False)
    .head(5)
)
st.bar_chart(top_cats[["Выручка", "Продажи"]])

# 3. ТОП-5 товаров по продажам
st.subheader("ТОП-5 товаров по продажам")
top5_goods = (
    filt_df.groupby("Товар")
    .agg({"Продажи": "sum", "Выручка": "sum"})
    .sort_values("Продажи", ascending=False)
    .head(5)
)
st.table(top5_goods.reset_index())

# 4. График возвратов и доли возвратов
agg["Доля возвратов, %"] = np.where(agg["Продажи"] > 0, agg["Возвраты"] / agg["Продажи"] * 100, 0)
st.subheader("Возвраты и их динамика")
st.bar_chart(agg.set_index("Дата")[["Возвраты", "Доля возвратов, %"]])

# 5. График расходов на маркетинг и конверсия
st.subheader("Расходы на маркетинг и конверсия")
st.line_chart(agg.set_index("Дата")[["Расходы на маркетинг", "Конверция"]])

# 6. График среднего рейтинга и среднего чека
st.subheader("Средний рейтинг и средний чек")
st.line_chart(agg.set_index("Дата")[["Средний рейтинг", "Средний чек"]])

# Данные со всеми товарами — только по желанию пользователя
show_data = st.checkbox("Показать все исходные данные по товарам")
if show_data:
    st.dataframe(filt_df.drop(columns=["Дата_dt"]).reset_index(drop=True))
