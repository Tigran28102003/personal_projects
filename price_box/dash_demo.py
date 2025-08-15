import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import random
import locale

# Установить локаль на русскую (если система это поддерживает)
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except:
    pass  # Если не удалось, продолжим без изменения локали

# ------------------- Генерация справочников -------------------

categories = [
    "Смартфоны", "Ноутбуки", "Планшеты", "Смарт-часы", "Наушники",
    "Телевизоры", "МФУ/Принтеры", "Мониторы", "Клавиатуры", "Мыши"
]

product_names = []
product_cats = []
for i in range(50):
    category = random.choice(categories)
    prod_name = f"{category} {chr(65 + i % 10)}-{1000 + i}"
    product_names.append(prod_name)
    product_cats.append(category)

# ------------------- Основные параметры и генерация -------------------

n_days = 365
start_date = pd.to_datetime("2024-01-01")
np.random.seed(42)
random.seed(42)

dates = pd.date_range(start=start_date, periods=n_days, freq='D')

base_sales = np.random.poisson(lam=12, size=(n_days, 50))
base_returns = np.random.binomial(base_sales, 0.12)
base_revenue = base_sales * np.random.randint(4000, 60000, size=(n_days, 50))
base_rating = np.random.beta(a=7, b=2, size=(n_days, 50)) * 4 + 1
base_marketing = np.random.gamma(shape=2, scale=400, size=(n_days, 50))
base_conversion = np.random.beta(a=2.5, b=12, size=(n_days, 50))

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

df["Средний чек"] = np.where(df["Продажи"] > 0, df["Выручка"] / df["Продажи"], 0)
df["Прибыль"] = df["Выручка"] - (df["Расходы на маркетинг"] + df["Возвраты"] * df["Выручка"].mean() * 0.1)
df["Дата_dt"] = pd.to_datetime(df["Дата"])

start_filter, end_filter = st.date_input(
    "Выберите временной период для анализа",
    value=[start_date, start_date + pd.Timedelta(days=n_days - 1)],
    min_value=start_date,
    max_value=start_date + pd.Timedelta(days=n_days - 1)
)

filt_df = df[(df["Дата_dt"] >= pd.to_datetime(start_filter)) & (df["Дата_dt"] <= pd.to_datetime(end_filter))]

agg = filt_df.groupby("Дата").agg({
    "Продажи": "sum",
    "Выручка": "sum",
    "Прибыль": "sum",
    "Возвраты": "sum",
    "Средний чек": "mean",
    "Средний рейтинг": "mean",
    "Расходы на маркетинг": "sum",
    "Конверсия": "mean"
}).reset_index()

items_sold = int(agg["Продажи"].sum())
total_revenue = agg["Выручка"].sum()
total_profit = agg["Прибыль"].sum()
total_returns = agg["Возвраты"].sum()
return_rate = (total_returns / items_sold * 100) if items_sold > 0 else 0
avg_check = agg["Средний чек"].mean()
avg_rating = agg["Средний рейтинг"].mean()
avg_conv = agg["Конверсия"].mean() * 100
total_marketing = agg["Расходы на маркетинг"].sum()

st.title("Дашборд селлера маркетплейса: техника (демо-данные)")

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Выручка, ₽", f"{total_revenue:,.0f}")
    col2.metric("Чистая прибыль, ₽", f"{total_profit:,.0f}")
    col3.metric("Продано товаров", f"{items_sold:,d}")
    col4.metric("Средний чек, ₽", f"{avg_check:,.0f}")

with st.container():
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Доля возвратов, %", f"{return_rate:.1f}")
    col6.metric("Средний рейтинг", f"{avg_rating:.2f}")
    col7.metric("Конверсия, %", f"{avg_conv:.2f}")
    col8.metric("Расходы на рекламу, ₽", f"{total_marketing:,.0f}")

color_revenue = "#1f77b4"   # оттенок синего для выручки
color_profit = "#4a90e2"    # светлее синий для прибыли
color_sales = "#003366"     # тёмно-синий для продаж

def plot_revenue_profit_sales(df):
    base = alt.Chart(df).encode(
        x=alt.X('Дата:T', axis=alt.Axis(labelAngle=45))
    )

    line_revenue = base.mark_line(color=color_revenue).encode(
        y=alt.Y('Выручка:Q', axis=alt.Axis(title='Выручка и Прибыль, ₽', titleColor=color_revenue))
    )
    line_profit = base.mark_line(color=color_profit).encode(
        y=alt.Y('Прибыль:Q', axis=None)
    )
    line_sales = base.mark_line(color=color_sales).encode(
        y=alt.Y('Продажи:Q', axis=alt.Axis(title='Продажи, шт.', titleColor=color_sales))
    )

    chart = alt.layer(line_revenue, line_profit, line_sales).resolve_scale(
        y='independent'
    ).properties(width=700, height=350)
    return chart

st.subheader("Динамика выручки, прибыли и продаж")
st.altair_chart(plot_revenue_profit_sales(agg), use_container_width=True)

agg["Доля возвратов, %"] = np.where(agg["Продажи"] > 0, agg["Возвраты"] / agg["Продажи"] * 100, 0)
st.subheader("Возвраты и их доля (%)")
st.altair_chart(
    alt.layer(
        alt.Chart(agg).mark_bar(color="#4a90e2").encode(
            x=alt.X('Дата:T', axis=alt.Axis(labelAngle=45)),
            y=alt.Y('Возвраты:Q', axis=alt.Axis(title='Количество возвратов', titleColor="#4a90e2"))
        ),
        alt.Chart(agg).mark_line(color="#003366").encode(
            x='Дата:T',
            y=alt.Y('Доля возвратов, %:Q', axis=alt.Axis(title='Доля возвратов, %', titleColor="#003366"))
        )
    ).resolve_scale(
        y='independent'
    ).properties(width=700, height=350),
    use_container_width=True
)

st.subheader("Расходы на маркетинг и конверсия")
st.altair_chart(
    alt.layer(
        alt.Chart(agg).mark_line(color="#1f77b4").encode(
            x=alt.X('Дата:T', axis=alt.Axis(labelAngle=45)),
            y=alt.Y('Расходы на маркетинг:Q', axis=alt.Axis(title='Расходы на маркетинг, ₽', titleColor="#1f77b4"))
        ),
        alt.Chart(agg).mark_line(color="#003366").encode(
            x='Дата:T',
            y=alt.Y('Конверсия:Q', axis=alt.Axis(title='Конверсия, %', titleColor="#003366"))
        )
    ).resolve_scale(
        y='independent'
    ).properties(width=700, height=350),
    use_container_width=True
)

# ТОП-5 категорий по выручке и продажам (перекрывающиеся диаграммы)
st.subheader("ТОП-5 категорий по выручке и продажам (выручка — столбцы, продажи — линия)")

top_cats = (
    filt_df.groupby("Категория")
    .agg({"Продажи": "sum", "Выручка": "sum"})
    .sort_values("Выручка", ascending=False)
    .head(5)
).reset_index()

base = alt.Chart(top_cats).encode(
    x=alt.X('Категория:N', sort='-y', axis=alt.Axis(title='Категории'))
)

bar_revenue = base.mark_bar(color='blue', opacity=0.5).encode(
    y=alt.Y('Выручка:Q', axis=alt.Axis(title='Выручка, ₽', titleColor='blue'))
)

line_sales_line = base.mark_line(color='darkblue', size=3).encode(
    y=alt.Y('Продажи:Q', axis=alt.Axis(title='Продажи, шт.', titleColor='darkblue'))
)

line_sales_points = base.mark_point(color='darkblue', size=50).encode(
    y='Продажи:Q'
)

chart = alt.layer(bar_revenue, line_sales_line, line_sales_points).resolve_scale(
    y='independent'
).properties(width=600, height=400)

st.altair_chart(chart, use_container_width=True)

# ТОП-5 товаров по продажам
st.subheader("ТОП-5 товаров по продажам")
top5_goods = (
    filt_df.groupby("Товар")
    .agg({"Продажи": "sum", "Выручка": "sum"})
    .sort_values("Продажи", ascending=False)
    .head(5)
)
st.table(top5_goods.reset_index())

# Средний рейтинг и средний чек
st.subheader("Средний рейтинг и средний чек")
st.altair_chart(
    alt.layer(
        alt.Chart(agg).mark_line(color=color_profit).encode(
            x=alt.X('Дата:T', axis=alt.Axis(labelAngle=45)),
            y=alt.Y('Средний рейтинг:Q', axis=alt.Axis(title='Средний рейтинг', titleColor=color_profit))
        ),
        alt.Chart(agg).mark_line(color=color_sales).encode(
            x='Дата:T',
            y=alt.Y('Средний чек:Q', axis=alt.Axis(title='Средний чек, ₽', titleColor=color_sales))
        )
    ).resolve_scale(
        y='independent'
    ).properties(width=700, height=350),
    use_container_width=True
)

# Отображение данных по желанию пользователя
show_data = st.checkbox("Показать все исходные данные по товарам")
if show_data:
    st.dataframe(filt_df.drop(columns=["Дата_dt"]).reset_index(drop=True))
