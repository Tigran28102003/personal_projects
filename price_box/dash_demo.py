import streamlit as st
import pandas as pd
import numpy as np

np.random.seed(42)

n_days = 360

start_date = pd.to_datetime("2024-01-01")
dates = pd.date_range(start=start_date, periods=n_days, freq='D')

# Продажи (пуассон)
sales = np.random.poisson(lam=20, size=n_days)

# Выручка (лог-нормальное)
revenue = np.random.lognormal(mean=4, sigma=0.3, size=n_days) * 100

# Возвраты (пуассон)
returns = np.random.poisson(lam=2, size=n_days)

# Рейтинг (бета, умноженная на 4 + 1)
ratings = np.random.beta(a=8, b=2, size=n_days) * 4 + 1

# Остатки (экспоненциальное)
stocks = np.maximum(0, 200 - np.cumsum(np.random.poisson(20, size=n_days)))

# Расходы на маркетинг (гамма)
marketing = np.random.gamma(2, 200, size=n_days)

# Конверсия (бета)
conversion = np.random.beta(3, 12, size=n_days)

profit = revenue - (marketing + returns*revenue.mean()*0.1)
items_sold = sales.sum()
total_revenue = revenue.sum()
total_profit = profit.sum()
total_returns = returns.sum()
return_rate = total_returns / items_sold * 100
avg_check = total_revenue / items_sold
avg_rating = ratings.mean()
avg_conv = conversion.mean() * 100
total_marketing = marketing.sum()

df = pd.DataFrame({
    "Дата": dates,
    "Продажи": sales,
    "Выручка": revenue,
    "Возвраты": returns,
    "Средний рейтинг": ratings,
    "Расходы на маркетинг": marketing,
    "Конверсия": conversion,
    "Прибыль": profit.round(2)
})

st.title("Демо-дашборд клиента")

# Карточки с ключевыми метриками
col1, col2, col3, col4 = st.columns(4)
col1.metric("Выручка, ₽", f"{total_revenue:,.0f}")
col2.metric("Чистая прибыль, ₽", f"{total_profit:,.0f}")
col3.metric("Продано товаров", items_sold)
col4.metric("Средний чек, ₽", f"{avg_check:,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Доля возвратов, %", f"{return_rate:.1f}")
col6.metric("Средний рейтинг", f"{avg_rating:.2f}")
col7.metric("Конверсия, %", f"{avg_conv:.2f}")
col8.metric("Расходы на рекламу, ₽", f"{total_marketing:,.0f}")

# Фильтр временного периода для графиков
start_filter, end_filter = st.date_input(
    "Выберите временной период для анализа",
    value=[start_date, start_date + pd.Timedelta(days=n_days-1)],
    min_value=start_date,
    max_value=start_date + pd.Timedelta(days=n_days-1)
)

# Фильтрация данных по выбранному диапазону
filtered_df = df[(df["Дата"] >= pd.to_datetime(start_filter)) & (df["Дата"] <= pd.to_datetime(end_filter))]

# Графики по фильтрованным данным
st.subheader("Продажи, Выручка, Возвраты, Остатки")
st.line_chart(filtered_df.set_index("Дата")[["Продажи", "Выручка", "Возвраты"]])

st.subheader("Расходы на маркетинг")
st.bar_chart(filtered_df.set_index("Дата")["Расходы на маркетинг"])

st.subheader("Средний рейтинг и Конверсия")
st.line_chart(filtered_df.set_index("Дата")[["Средний рейтинг", "Конверсия"]])

# Таблица с отфильтрованными данными
st.dataframe(filtered_df.reset_index(drop=True))
