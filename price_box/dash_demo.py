import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import random
import locale
from datetime import datetime, timedelta

# Установить локаль на русскую
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except:
    pass

# ------------------- Кэширование данных -------------------
@st.cache_data  # Кэшируем данные для ускорения работы приложения
def generate_data():
    categories = [
        "Смартфоны", "Ноутбуки", "Планшеты", "Смарт-часы", "Наушники",
        "Телевизоры", "МФУ/Принтеры", "Мониторы", "Клавиатуры", "Мыши"
    ]

    # Генерация данных
    product_names = []
    product_cats = []
    for i in range(50):
        category = random.choice(categories)
        prod_name = f"{category} {chr(65 + i % 10)}-{1000 + i}"
        product_names.append(prod_name)
        product_cats.append(category)

    n_days = 365
    start_date = pd.to_datetime("2024-01-01")
    np.random.seed(42)
    random.seed(42)

    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    # Генерация случайных данных с более реалистичными паттернами
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
    df["ROI"] = df["Выручка"] / df["Расходы на маркетинг"]  # Добавили ROI

    return df, categories

df, categories = generate_data()

# ------------------- Фильтры -------------------
st.sidebar.header("Параметры анализа")

# Фильтр по дате с возможностью выбора предустановленных периодов
date_options = {
    "Последние 30 дней": (datetime.now() - timedelta(days=30), datetime.now()),
    "Последние 90 дней": (datetime.now() - timedelta(days=90), datetime.now()),
    "Последний год": (datetime.now() - timedelta(days=365), datetime.now()),
    "Произвольный период": None
}

selected_period = st.sidebar.selectbox(
    "Выберите период",
    list(date_options.keys()),
    index=2
)

if date_options[selected_period]:
    start_filter, end_filter = date_options[selected_period]
else:
    start_filter, end_filter = st.sidebar.date_input(
        "Выберите временной период",
        value=[df["Дата_dt"].min(), df["Дата_dt"].max()],
        min_value=df["Дата_dt"].min(),
        max_value=df["Дата_dt"].max()
    )

# Фильтр по категориям (мультиселект)
selected_categories = st.sidebar.multiselect(
    "Выберите категории товаров",
    options=categories,
    default=categories,
    help="Позволяет анализировать отдельные категории или их комбинации"
)

# Применение фильтров
filt_df = df[
    (df["Дата_dt"] >= pd.to_datetime(start_filter)) &
    (df["Дата_dt"] <= pd.to_datetime(end_filter)) &
    (df["Категория"].isin(selected_categories))
]

# ------------------- Основные метрики -------------------
st.title("📊 Дашборд селлера маркетплейса: техника (демо-данные)")

# Агрегация данных с учетом фильтров
agg = filt_df.groupby("Дата").agg({
    "Продажи": "sum",
    "Выручка": "sum",
    "Прибыль": "sum",
    "Возвраты": "sum",
    "Средний чек": "mean",
    "Средний рейтинг": "mean",
    "Расходы на маркетинг": "sum",
    "Конверсия": "mean",
    "ROI": "mean"
}).reset_index()

# Расчет ключевых показателей
items_sold = int(agg["Продажи"].sum())
total_revenue = agg["Выручка"].sum()
total_profit = agg["Прибыль"].sum()
total_returns = agg["Возвраты"].sum()
return_rate = (total_returns / items_sold * 100) if items_sold > 0 else 0
avg_check = agg["Средний чек"].mean()
avg_rating = agg["Средний рейтинг"].mean()
avg_conv = agg["Конверсия"].mean() * 100
total_marketing = agg["Расходы на маркетинг"].sum()
avg_roi = agg["ROI"].mean()

# Отображение метрик в колонках с иконками и delta-значениями
with st.container():
    cols = st.columns(4)
    cols[0].metric("💰 Выручка", f"{total_revenue:,.0f} ₽",
                  help="Общая выручка за период")
    cols[1].metric("💵 Чистая прибыль", f"{total_profit:,.0f} ₽",
                  delta=f"{total_profit/total_revenue*100:.1f}% маржа" if total_revenue > 0 else 0,
                  help="Прибыль после вычета расходов и возвратов")
    cols[2].metric("🛒 Продано товаров", f"{items_sold:,d}",
                  help="Общее количество проданных единиц товара")
    cols[3].metric("🧾 Средний чек", f"{avg_check:,.0f} ₽",
                  help="Средняя сумма одного заказа")

with st.container():
    cols = st.columns(4)
    cols[0].metric("🔄 Доля возвратов", f"{return_rate:.1f}%",
                  delta_color="inverse",
                  help="Процент возвращенных товаров от общего числа продаж")
    cols[1].metric("⭐ Средний рейтинг", f"{avg_rating:.1f}/5",
                  help="Средняя оценка товаров покупателями")
    cols[2].metric("📈 Конверсия", f"{avg_conv:.1f}%",
                  help="Процент посетителей, совершивших покупку")
    cols[3].metric("📊 ROI маркетинга", f"{avg_roi:.1f}x",
                  help="Окупаемость маркетинговых инвестиций")

# ------------------- Улучшенные визуализации -------------------

# 1. График выручки, прибыли и продаж с area chart и tooltips
st.subheader("📈 Динамика ключевых показателей")
def plot_enhanced_revenue_profit_sales(df):
    base = alt.Chart(df).encode(
        x=alt.X('Дата:T', axis=alt.Axis(labelAngle=-45, title=None))
    )

    # Area chart для выручки с градиентной заливкой
    area_revenue = base.mark_area(
        line={'color':'#1f77b4'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='#1f77b4', offset=1)],
            x1=0, x2=0, y1=1, y2=0
        ),
        opacity=0.6,
        interpolate='monotone'
    ).encode(
        y=alt.Y('Выручка:Q', axis=alt.Axis(title='Выручка, ₽', titleColor='#1f77b4')),
        tooltip=['Дата:T', 'Выручка:Q', 'Прибыль:Q', 'Продажи:Q']
    )

    # Линия для прибыли
    line_profit = base.mark_line(color='#4a90e2', strokeWidth=2).encode(
        y=alt.Y('Прибыль:Q'),
        tooltip=['Прибыль:Q']
    )

    # Линия для продаж на второй оси
    line_sales = base.mark_line(color='#003366', strokeWidth=2).encode(
        y=alt.Y('Продажи:Q', axis=alt.Axis(title='Продажи, шт.', titleColor='#003366')),
        tooltip=['Продажи:Q']
    )

    # Комбинированный график
    chart = alt.layer(
        area_revenue,
        line_profit,
        line_sales
    ).resolve_scale(
        y='independent'
    ).properties(
        height=400
    ).interactive()

    return chart

st.altair_chart(plot_enhanced_revenue_profit_sales(agg), use_container_width=True)

# 2. Улучшенный график возвратов с областью и средней линией
st.subheader("🔄 Анализ возвратов")
agg["Доля возвратов, %"] = np.where(agg["Продажи"] > 0, agg["Возвраты"] / agg["Продажи"] * 100, 0)

return_chart = alt.Chart(agg).mark_bar(color='#4a90e2', opacity=0.7).encode(
    x=alt.X('Дата:T', axis=alt.Axis(labelAngle=-45, title=None)),
    y=alt.Y('Возвраты:Q', axis=alt.Axis(title='Количество возвратов', titleColor='#4a90e2')),
    tooltip=['Дата:T', 'Возвраты:Q', 'Доля возвратов, %:Q']
)

return_rate_area = alt.Chart(agg).mark_area(
    color='#003366',
    opacity=0.3,
    interpolate='monotone'
).encode(
    x='Дата:T',
    y=alt.Y('Доля возвратов, %:Q', axis=alt.Axis(title='Доля возвратов, %', titleColor='#003366'))
)

mean_return_rate = alt.Chart(agg).mark_rule(
    color='red',
    strokeDash=[5,5]
).encode(
    y='mean(Доля возвратов, %):Q',
    size=alt.value(1),
    tooltip=[alt.Tooltip('mean(Доля возвратов, %):Q', title='Средняя доля возвратов')]
)

st.altair_chart(
    (return_chart + return_rate_area + mean_return_rate).resolve_scale(
        y='independent'
    ).properties(
        height=350
    ).interactive(),
    use_container_width=True
)

# 3. График маркетинга и конверсии с ROI
st.subheader("📢 Эффективность маркетинга")
tab1, tab2 = st.tabs(["Расходы и конверсия", "ROI"])

with tab1:
    marketing_chart = alt.Chart(agg).mark_area(
        color='#1f77b4',
        opacity=0.5,
        interpolate='monotone'
    ).encode(
        x=alt.X('Дата:T', axis=alt.Axis(labelAngle=-45, title=None)),
        y=alt.Y('Расходы на маркетинг:Q', axis=alt.Axis(title='Расходы на маркетинг, ₽', titleColor='#1f77b4')),
        tooltip=['Дата:T', 'Расходы на маркетинг:Q']
    )

    conversion_line = alt.Chart(agg).mark_line(
        color='#003366',
        strokeWidth=2
    ).encode(
        x='Дата:T',
        y=alt.Y('Конверсия:Q', axis=alt.Axis(title='Конверсия, %', titleColor='#003366')),
        tooltip=['Конверсия:Q']
    )

    st.altair_chart(
        (marketing_chart + conversion_line).resolve_scale(
            y='independent'
        ).properties(
            height=350
        ).interactive(),
        use_container_width=True
    )

with tab2:
    roi_chart = alt.Chart(agg).mark_line(
        color='green',
        strokeWidth=3,
        interpolate='monotone'
    ).encode(
        x=alt.X('Дата:T', axis=alt.Axis(labelAngle=-45, title=None)),
        y=alt.Y('ROI:Q', axis=alt.Axis(title='ROI (выручка/расходы)', titleColor='green')),
        tooltip=['Дата:T', 'ROI:Q']
    )

    st.altair_chart(roi_chart.properties(height=350).interactive(), use_container_width=True)

# 4. ABC-анализ товаров
st.subheader("📊 ABC-анализ товаров")
abc_tab1, abc_tab2 = st.tabs(["По выручке", "По количеству продаж"])

with abc_tab1:
    top_goods_revenue = (
        filt_df.groupby(["Категория", "Товар"])
        .agg({"Выручка": "sum", "Продажи": "sum"})
        .sort_values("Выручка", ascending=False)
        .reset_index()
    )

    # Добавляем кумулятивную долю для ABC-анализа
    top_goods_revenue['Доля'] = top_goods_revenue['Выручка'] / top_goods_revenue['Выручка'].sum()
    top_goods_revenue['Кумулятивная доля'] = top_goods_revenue['Доля'].cumsum()

    abc_chart = alt.Chart(top_goods_revenue.head(20)).mark_bar().encode(
        x=alt.X('Товар:N', sort='-y', title="Товар"),
        y=alt.Y('Выручка:Q', title="Выручка, ₽"),
        color=alt.Color('Категория:N', legend=alt.Legend(title="Категория")),
        tooltip=['Товар', 'Категория', 'Выручка:Q', 'Продажи:Q']
    ).properties(
        height=400
    )

    st.altair_chart(abc_chart, use_container_width=True)

with abc_tab2:
    top_goods_sales = (
        filt_df.groupby(["Категория", "Товар"])
        .agg({"Продажи": "sum", "Выручка": "sum"})
        .sort_values("Продажи", ascending=False)
        .reset_index()
    )

    sales_chart = alt.Chart(top_goods_sales.head(20)).mark_bar().encode(
        x=alt.X('Товар:N', sort='-y', title="Товар"),
        y=alt.Y('Продажи:Q', title="Количество продаж"),
        color=alt.Color('Категория:N', legend=None),
        tooltip=['Товар', 'Категория', 'Продажи:Q', 'Выручка:Q']
    ).properties(
        height=400
    )

    st.altair_chart(sales_chart, use_container_width=True)

# 5. Анализ по дням недели
st.subheader("📅 Анализ по дням недели")
# Преобразуем столбец 'Дата' в datetime, если это еще не сделано
agg['Дата'] = pd.to_datetime(agg['Дата'])

# Теперь можно извлечь день недели
agg['День недели'] = agg['Дата'].dt.day_name()

# # Для русской локализации можно использовать (если локаль установлена правильно):
# agg['День недели'] = agg['Дата'].dt.strftime('%A')

weekday_analysis = agg.groupby('День недели').agg({
    'Выручка': 'mean',
    'Продажи': 'mean',
    'Конверсия': 'mean'
}).reset_index()

# Упорядочиваем дни недели
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_analysis['День недели'] = pd.Categorical(
    weekday_analysis['День недели'],
    categories=weekday_order,
    ordered=True
)
weekday_analysis = weekday_analysis.sort_values('День недели')

weekday_chart = alt.Chart(weekday_analysis).mark_bar().encode(
    x=alt.X('День недели:N', title=None),
    y=alt.Y('Выручка:Q', title="Средняя выручка, ₽"),
    color=alt.Color('День недели:N', legend=None),
    tooltip=['День недели', 'Выручка:Q', 'Продажи:Q']
).properties(
    height=350
)

st.altair_chart(weekday_chart, use_container_width=True)

# ------------------- Дополнительные функции -------------------
expander = st.expander("🔍 Детализированные данные")
with expander:
    st.write("### Полные данные за выбранный период")
    st.dataframe(
        filt_df.drop(columns=["Дата_dt"]).sort_values("Дата", ascending=False),
        height=300,
        use_container_width=True
    )

    # Кнопка для скачивания данных
    csv = filt_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать данные в CSV",
        data=csv,
        file_name='ozon_sales_data.csv',
        mime='text/csv'
    )
