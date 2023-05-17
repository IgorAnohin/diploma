from pathlib import Path
from typing import List, Tuple
import plotly.graph_objs as go
from keras import backend as K

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.losses import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from diploma.models.base import MLModel


def _create_lstm_dataset(dataset, look_back=5) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []

    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def _create_lstm_dataset_with_future(dataset, look_back=5, predict_forward=0) -> Tuple[
    np.ndarray, np.ndarray]:
    X, Y = [], []

    for i in range(len(dataset) - look_back - predict_forward):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back + predict_forward, 0])
    return np.array(X), np.array(Y)


def train(csv_path: str, epochs: int = 5) -> MLModel:
    # Загрузка и подготовка данных
    data = pd.read_csv(csv_path)
    close_prices = data['Close'].values.reshape(-1, 1)

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler.fit_transform(close_prices)

    close_prices = close_prices[:len(close_prices) // 2]

    look_back = 5
    # Создание набора данных для обучения
    X, Y = _create_lstm_dataset(close_prices, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Разделение данных на обучающий и тестовый наборы
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Создание и обучение LSTM-модели
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=500, verbose=1)

    # Предсказание и оценка модели
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    # Обратное преобразование значений
    Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
    Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
    Y_pred_train = scaler.inverse_transform(Y_pred_train)
    Y_pred_test = scaler.inverse_transform(Y_pred_test)

    # Вычисление ошибки
    train_score = mean_squared_error(Y_train, Y_pred_train)
    test_score = mean_squared_error(Y_test, Y_pred_test)
    print(f'Train RMSE: {sum(train_score) / len(train_score)}')
    print(f'Test RMSE: {sum(test_score) / len(test_score)}')
    r2 = r2_score(Y_test, Y_pred_test)
    print(f"R-квадрат: {r2}")

    # plotting the points
    # plt.plot([idx for idx in range(len(Y_test))], Y_test)
    # plt.plot([idx for idx in range(len(Y_pred_test))], Y_pred_test)
    #
    # # function to show the plot
    # plt.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[idx for idx in range(len(Y_train))], y=Y_train.flatten(), name='Train'))
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))], y=Y_pred_train.flatten(),
                             name='Train Predicted'))
    fig.add_trace(
        go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))], y=Y_test.flatten(),
                   name='Test'))
    fig.add_trace(
        go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))], y=Y_pred_test.flatten(),
                   name='Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    return MLModel(model, scaler, required_prices=5, predicts_forward=1)


def train_complex(csv_path: Path, epochs: int = 80) -> MLModel:
    # Загрузка и подготовка данных
    data = pd.read_csv(csv_path)
    close_prices = data['Close'].values.reshape(-1, 1)

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))

    close_prices = scaler.fit_transform(close_prices)

    close_prices = close_prices[:len(close_prices) // 2]

    look_back = 40
    predict_forward = 10
    # Создание набора данных для обучения
    X, Y = _create_lstm_dataset_with_future(close_prices, look_back,
                                            predict_forward=predict_forward)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Разделение данных на обучающий и тестовый наборы
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Создание и обучение LSTM-модели
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(75))
    model.add(Dropout(0.2))
    model.add(Dense(20))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model_save_file = Path(f"tmp/{csv_path.name.split('_')[0]}_lstm_complex.tf")
    if model_save_file.exists():
        print(f"Upload model weights from {model_save_file}")
        model.load_weights(model_save_file)
    else:
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,
                  batch_size=4000, verbose=1, callbacks=[early_stop])

        # Предсказание и оценка модели
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        # Обратное преобразование значений
        Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
        Y_pred_train = scaler.inverse_transform(Y_pred_train)
        Y_pred_test = scaler.inverse_transform(Y_pred_test)

        # Вычисление ошибки
        train_score = mean_squared_error(Y_train, Y_pred_train)
        test_score = mean_squared_error(Y_test, Y_pred_test)
        print(f'Train RMSE: {sum(train_score) / len(train_score)}')
        print(f'Test RMSE: {sum(test_score) / len(test_score)}')
        r2 = r2_score(Y_test, Y_pred_test)
        print(f"R-квадрат: {r2}")

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))], y=Y_train.flatten(),
        # name='Train'))
        # fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))],
        # y=Y_pred_train.flatten(), name='Train Predicted'))
        # fig.add_trace(go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))],
        # y=Y_test.flatten(), name='Test'))
        # fig.add_trace(go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))],
        # y=Y_pred_test.flatten(), name='Predicted'))
        # fig.update_layout(title=f'Test and Predicted Values {model_save_file}',
        # xaxis_title='Date', yaxis_title='Price')
        # fig.show()

        model.save(model_save_file)

    return MLModel(model, scaler, required_prices=look_back, predicts_forward=predict_forward)


def train_with_returns(csv_path: str):
    # Загрузка и подготовка данных
    data = pd.read_csv(csv_path)

    # close_prices = data['Close'].values.reshape(-1, 1)
    time_series = data['Close'] / data['Close'].shift(1)
    time_series = time_series.fillna(1).values.reshape(-1, 1)

    look_back = 30
    # Создание набора данных для обучения
    X, Y = _create_lstm_dataset_with_future(time_series, look_back, predict_forward=10)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Разделение данных на обучающий и тестовый наборы
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Создание и обучение LSTM-модели
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=5, batch_size=500, verbose=1)

    # Предсказание и оценка модели
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    # Вычисление ошибки
    # train_score = mean_squared_error(Y_train, Y_pred_train)
    test_score = mean_squared_error(Y_test, Y_pred_test)
    # print(f'Train RMSE: {sum(train_score) / len(train_score)}')
    print(f'Test RMSE: {sum(test_score) / len(test_score)}')

    r2 = r2_score(Y_test, Y_pred_test)
    print(f"R-квадрат: {r2}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[idx for idx in range(len(Y_train))], y=Y_train.flatten(), name='Train'))
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))], y=Y_pred_train.flatten(),
                             name='Train Predicted'))
    fig.add_trace(
        go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))], y=Y_test.flatten(),
                   name='Test'))
    fig.add_trace(
        go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_test))], y=Y_pred_test.flatten(),
                   name='Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[idx for idx in range(len(data["Close"]))], y=data["Close"], name='real'))
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))],
                             y=convert_returns_to_prices_adj(data["Close"], Y_train.flatten()),
                             name='Train'))
    fig.add_trace(go.Scatter(x=[idx for idx in range(len(Y_train))],
                             y=convert_returns_to_prices_adj(data["Close"],
                                                             Y_pred_train.flatten()),
                             name='Train Predicted'))
    fig.add_trace(go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_pred_test))],
                             y=convert_returns_to_prices_adj(data["Close"][train_size:],
                                                             Y_test.flatten()), name='Test'))
    fig.add_trace(go.Scatter(x=[idx + len(Y_train) for idx in range(len(Y_pred_test))],
                             y=convert_returns_to_prices_adj(data["Close"][train_size:],
                                                             Y_pred_test.flatten()),
                             name='Test Predicted'))
    fig.update_layout(title='Test and Predicted Values', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    # return MLModel(model, scaler, required_prices=5)


def convert_returns_to_prices_adj(close_prices: List[float], returns: List[float]) -> List[float]:
    result = []
    for prev_price, return_diff in zip(close_prices, returns):
        result.append(prev_price * return_diff)

    return result


def convert_returns_to_prices_auto(initial_price: float, returns: List[float]) -> List[float]:
    prev_price = initial_price

    result = []
    for return_diff in returns:
        prev_price *= return_diff
        result.append(prev_price)

    return result


def cache_second_half(model: MLModel, csv_path: Path) -> None:
    # Загрузка и подготовка данных
    data = pd.read_csv(csv_path)
    close_prices = data['Close'].values.reshape(-1, 1)

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler.fit_transform(close_prices)

    cache_file_path = Path(f"tmp/cache_{csv_path.name}")
    if cache_file_path.exists():
        print(f"Load cache from {cache_file_path}")
        model.load_cache(cache_file_path)
    else:
        # first_half_close_prices = close_prices[:len(close_prices) // 2]
        # second_half_close_prices = close_prices[len(close_prices) // 2:]

        # Создание набора данных для обучения
        # Создание набора данных для обучения
        # X_train, Y_train = _create_lstm_dataset_with_future(first_half_close_prices,
        # model.loop_back,
        #                                         predict_forward=model.predicts_forward)
        # X, Y = _create_lstm_dataset_with_future(second_half_close_prices, model.loop_back,
        #                                         predict_forward=model.predicts_forward)
        # X_train_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
        # X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X, Y = _create_lstm_dataset_with_future(close_prices, model.loop_back,
                                                predict_forward=model.predicts_forward)
        X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model.create_cache(
            X_reshaped[:len(X_reshaped) // 2], Y[:len(X_reshaped) // 2],
            X_reshaped[len(X_reshaped) // 2:], Y[len(X_reshaped) // 2:],
        )

        model.save_cache(cache_file_path)

    # half_prices = len(close_prices) // 2
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=[idx for idx in range(half_prices)],
    #                          y=close_prices[half_prices:].flatten(), name='Real prices'))
    # fig.add_trace(go.Scatter(
    #     x=[idx + model.loop_back + model.loop_back for idx in range(len(model._cache.values()))],
    #     y=list(model._cache.values()), name='Train Predicted'))
    # fig.update_layout(title=f'Test and Predicted Values {csv_path}', xaxis_title='Date',
    #                   yaxis_title='Price')
    # fig.show()
