from concurrent.futures import ProcessPoolExecutor
from copy import copy

import plotly.graph_objs as go

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# Prepare the data for the sliding window approach
def create_sliding_window(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_horizon])
    return np.array(X), np.array(y)


def train(csv_path: str):
    # Чтение данных
    data = pd.read_csv(csv_path)

    # Use the 'Close' column or any other desired column for forecasting
    time_series = data['Close']

    # Define the training and testing sets
    train_size = int(len(time_series) * 0.8)
    train, test = time_series[:train_size], time_series[train_size:]

    window_size = 5
    forecast_horizon = 1

    X_train, y_train = create_sliding_window(train, window_size, forecast_horizon)
    X_test, y_test = create_sliding_window(test, window_size, forecast_horizon)

    # Создание и обучение модели CatBoost
    model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=10, loss_function='RMSE')
    model.fit(X_train, y_train.ravel())

    y_train_pred = model.predict(X_train)
    # Предсказание
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: {:.4f}".format(mse))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, name='Train'))
    fig.add_trace(go.Scatter(x=train.index, y=y_train_pred, name='Train Predicted'))
    fig.add_trace(go.Scatter(x=test.index, y=test, name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=y_pred, name='Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()


def _train_and_predict(X_train, y_train, X_test):
    # Создание и обучение модели CatBoost
    model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=10, loss_function='RMSE')
    model.fit(X_train, y_train.ravel())

    return model.predict(X_test)


def train_with_batches(csv_path: str):
    # Чтение данных
    data = pd.read_csv(csv_path)

    # Use the 'Close' column or any other desired column for forecasting
    time_series = data['Close']

    # Define the training and testing sets
    train_size = int(len(time_series) * 0.8)
    start_offset = int(len(time_series) * 0.5)
    train, test = time_series[start_offset:train_size], time_series[train_size:]

    window_size = 5
    forecast_horizon = 1

    X_train, y_train = create_sliding_window(train, window_size, forecast_horizon)
    X_test, y_test = create_sliding_window(test, window_size, forecast_horizon)

    prediction_batch = 2000
    N_test_observations = len(test) // prediction_batch
    model_predictions = []

    futures = []
    X_history = X_train
    y_history = y_train
    with ProcessPoolExecutor(max_workers=20) as executor:
        for time_point in tqdm(range(N_test_observations)):
            test_values = X_test[time_point * prediction_batch:time_point * prediction_batch + prediction_batch]

            predict_future = executor.submit(_train_and_predict, copy(X_history), copy(y_history), test_values)
            futures.append(predict_future)

            X_history = np.vstack((X_history, test_values))
            y_history = np.vstack((y_history, y_test[time_point * prediction_batch:time_point * prediction_batch + prediction_batch]))

        for time_point in tqdm(range(N_test_observations)):
            result = futures[time_point].result()
            model_predictions.extend(result)

    # Оценка модели
    mse = mean_squared_error(y_test[:len(model_predictions)], model_predictions)
    print("Mean Squared Error: {:.4f}".format(mse))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(model_predictions))], y=test[:len(model_predictions)], name='Test'))
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(model_predictions))], y=model_predictions, name='Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()


def train_returns(csv_path: str):
    # Чтение данных
    data = pd.read_csv(csv_path)

    # Use the 'Close' column or any other desired column for forecasting
    # time_series = data['Close'].diff().fillna(0)
    time_series = data['Close'] / data['Close'].shift(1)

    # Define the training and testing sets
    train_size = int(len(time_series) * 0.8)
    train, test = time_series[:train_size], time_series[train_size:]

    window_size = 15
    forecast_horizon = 1

    X_train, y_train = create_sliding_window(train, window_size, forecast_horizon)
    X_test, y_test = create_sliding_window(test, window_size, forecast_horizon)

    # Создание и обучение модели CatBoost
    model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=10, loss_function='RMSE')
    model.fit(X_train, y_train.ravel())

    y_train_pred = model.predict(X_train)
    # Предсказание
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: {:.4f}".format(mse))

    start_price = data['Close'][train_size]
    predicted_prices = []

    prev_price = start_price
    for diff in y_pred:
        prev_price = prev_price * diff
        predicted_prices.append(prev_price)

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=train.index, y=train, name='Train'))
    # fig.add_trace(go.Scatter(x=train.index, y=y_train_pred, name='Train Predicted'))
    fig.add_trace(go.Scatter(x=test.index, y=data['Close'][train_size:], name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=predicted_prices, name='Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()
