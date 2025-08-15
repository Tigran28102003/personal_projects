import streamlit as st
import pandas as pd
import numpy as np

np.random.seed(42)

import streamlit as st
import numpy as np
import pandas as pd

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

st.title("Дашборд селлера маркетплейса (прототип, демо-данные)")

st.line_chart(df.set_index("День")[["Продажи", "Выручка", "Возвраты", "Остатки на складе"]])
st.bar_chart(df.set_index("День")["Расходы на маркетинг"])
st.line_chart(df.set_index("День")[["Средний рейтинг", "CR (конверсия)"]])

st.dataframe(df)
