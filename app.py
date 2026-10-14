import random
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

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
SEED = 42


def create_sequences(data, window_size=WINDOW):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


if st.button("Tải dữ liệu & Chạy dự báo", type="primary"):
    # Cố định seed để kết quả ổn định
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    with st.spinner("Đang tải dữ liệu..."):
        df = load_data(ticker.strip(), start_date, end_date)

    if df.empty or len(df) < 200:
        st.error("Không đủ dữ liệu cho mã cổ phiếu / khoảng thời gian này (cần ít nhất ~200 phiên). Hãy thử lại.")
        st.stop()

    st.subheader(f"Dữ liệu giá của {ticker.upper()}")
    st.dataframe(df.tail(10), use_container_width=True)

    data = df[["Close"]].values

    # Chia train / test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-WINDOW:], test_scaled))
    X_test, y_test = create_sequences(test_combined)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Tập validation = 10% cuối của train
    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]

    # Mô hình LSTM
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    with st.spinner("Đang huấn luyện mô hình LSTM (có thể mất vài phút)..."):
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0,
        )

    # Đánh giá
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
    mae = mean_absolute_error(y_test_real, test_pred)
    r2 = r2_score(y_test_real, test_pred)

    st.subheader("Đánh giá mô hình trên tập test")
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:,.2f}")
    m2.metric("MAE", f"{mae:,.2f}")
    m3.metric("R²", f"{r2:.4f}")

    # Biểu đồ loss
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Train vs Validation")
    ax1.legend()
    st.pyplot(fig1)

    # Biểu đồ giá thật vs dự báo
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(y_test_real, label="Giá thật")
    ax2.plot(test_pred, label="Dự báo")
    ax2.set_title("Giá thật vs Dự báo (tập test)")
    ax2.legend()
    st.pyplot(fig2)

    # Dự báo n ngày tới
    last_window = np.vstack((train_scaled, test_scaled))[-WINDOW:].reshape(1, WINDOW, 1)
    future_predictions = []
    for _ in range(int(n_days)):
        next_pred = model.predict(last_window, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_window = np.append(last_window[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=int(n_days))

    st.subheader(f"Dự báo {n_days} ngày tới")
    st.dataframe(
        pd.DataFrame({"Ngày": future_dates.date, "Giá dự báo": future_prices.round(2)}),
        use_container_width=True,
    )

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(y_test_real, label="Giá thật")
    ax3.plot(test_pred, label="Dự báo test")
    future_x = range(len(y_test_real), len(y_test_real) + int(n_days))
    ax3.plot(future_x, future_prices, marker="o", label=f"{n_days} ngày tới")
    ax3.set_title("Dự báo tương lai")
    ax3.legend()
    st.pyplot(fig3)

    st.caption("⚠️ Kết quả chỉ mang tính tham khảo, không phải lời khuyên đầu tư.")
