import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

st.set_page_config(page_title="Dự báo giá cổ phiếu bằng AI (LSTM)", page_icon="📈", layout="wide")

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", value="VNM.VN")

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input("📅 Chọn ngày bắt đầu:", value=date(2018, 1, 1))
with col2:
    end_date = st.date_input("📅 Chọn ngày kết thúc:", value=date.today())
with col3:
    n_days = st.number_input("🔮 Số ngày muốn dự báo:", min_value=1, max_value=60, value=10, step=1)

WINDOW = 60


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_model(window):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


if st.button("Tải dữ liệu & Chạy dự báo", type="primary"):
    with st.spinner("Đang tải dữ liệu..."):
        df = load_data(ticker.strip(), start_date, end_date)

    if df.empty or len(df) <= WINDOW + 10:
        st.error("Không đủ dữ liệu cho mã cổ phiếu / khoảng thời gian này. Hãy thử lại.")
        st.stop()

    st.subheader(f"Dữ liệu giá của {ticker.upper()}")
    st.dataframe(df.tail(10), use_container_width=True)

    close = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i - WINDOW:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with st.spinner("Đang huấn luyện mô hình LSTM (có thể mất vài phút)..."):
        model = build_model(WINDOW)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    pred_test = scaler.inverse_transform(model.predict(X_test, verbose=0))
    real_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = float(np.sqrt(np.mean((pred_test - real_test) ** 2)))
    mape = float(np.mean(np.abs((real_test - pred_test) / real_test)) * 100)

    m1, m2 = st.columns(2)
    m1.metric("RMSE", f"{rmse:,.2f}")
    m2.metric("MAPE", f"{mape:.2f}%")

    last_window = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    future = []
    for _ in range(int(n_days)):
        nxt = model.predict(last_window, verbose=0)[0, 0]
        future.append(nxt)
        last_window = np.append(last_window[:, 1:, :], [[[nxt]]], axis=1)
    future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=int(n_days))

    forecast_df = pd.DataFrame({"Ngày": future_dates.date, "Giá dự báo": future_prices.round(2)})

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Giá thực tế")
    test_index = df.index[WINDOW + split:]
    ax.plot(test_index, pred_test.flatten(), label="Dự đoán (tập test)", alpha=0.8)
    ax.plot(future_dates, future_prices, "r--o", label=f"Dự báo {n_days} ngày")
    ax.set_title(f"Dự báo giá cổ phiếu {ticker.upper()}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader(f"Bảng dự báo {n_days} ngày tới")
    st.dataframe(forecast_df, use_container_width=True)

    st.caption("⚠️ Kết quả chỉ mang tính tham khảo, không phải lời khuyên đầu tư.")
    streamlit
yfinance
numpy
pandas
matplotlib
scikit-learn
tensorflow-cpu
