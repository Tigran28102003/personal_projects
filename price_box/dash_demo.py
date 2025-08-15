import streamlit as st
import pandas as pd
import numpy as np

np.random.seed(42)

n_days = 360

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
    "День": range(1, n_days+1),
    "Продажи": sales,
    "Выручка": revenue,
    "Возвраты": returns,
    "Средний рейтинг": ratings.round(2),
    "Остатки на складе": stocks,
    "Расходы на маркетинг": marketing.round(2),
    "CR (конверсия)": conversion.round(2)
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


st.line_chart(df.set_index("День")[["Продажи", "Выручка", "Возвраты", "Остатки на складе"]])
st.bar_chart(df.set_index("День")["Расходы на маркетинг"])
st.line_chart(df.set_index("День")[["Средний рейтинг", "CR (конверсия)"]])

st.dataframe(df)
