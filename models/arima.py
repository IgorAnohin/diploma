import plotly.graph_objs as go
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from diploma.models.base import MLModel


def train(first_half_csv_path: str) -> None:
    # Загрузка данных из CSV-файла
    data = pd.read_csv(first_half_csv_path, parse_dates=["Time"])
    # data = data.set_index("Time")

    # Используем столбец Close как целевую переменную для предсказания
    target = data["Close"]

    # Разделение данных на обучающую и тестовую выборки
    train_size = int(len(target) * 0.9)
    start_offset = int(len(target) * 0.7)
    train, test = target[start_offset:train_size], target[train_size:]

    # Создание и обучение модели ARIMA
    model = ARIMA(train, order=(10, 0, 0))
    # model = ARIMA(train, order=(4, 1, 5))
    model_fit = model.fit()


    y_train_pred = model_fit.predict(start=0, end=len(train) - 1)

    # Предсказание на тестовой выборке
    y_pred = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
    print(y_pred)

    # Вычисление среднеквадратической ошибки
    mse = mean_squared_error(test, y_pred)
    print("Mean Squared Error: {:.4f}".format(mse))

    # # plotting the points
    # plt.plot([idx for idx in range(len(test))], test)
    # plt.plot([idx for idx in range(len(y_pred))], y_pred, color="green")
    #
    # # function to show the plot
    # plt.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, name='Train'))
    fig.add_trace(go.Scatter(x=train.index, y=y_train_pred, name='Train Predicted'))
    fig.add_trace(go.Scatter(x=test.index, y=test, name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=y_pred, name='Predicted'))
    fig.update_layout(title='ARIMA', xaxis_title='Date', yaxis_title='Price')
    fig.show()


def train_auto(first_half_csv_path: str) -> None:
    # Загрузка данных из CSV-файла
    data = pd.read_csv(first_half_csv_path)
    # data = data.set_index("Time")

    # Используем столбец Close как целевую переменную для предсказания
    target = data["Close"]
    # target = np.log(data['Close']).diff().dropna()

    # Разделение данных на обучающую и тестовую выборки
    train_size = int(len(target) * 0.95)
    start_offset = int(len(target) * 0.8)
    train, test = target[start_offset:train_size], target[train_size:]

    # Создание и обучение модели ARIMA
    model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
    print(model.summary())

    # Fit the ARIMA model
    model.fit(train)

    # Evaluate the model's performance on the test set
    predictions = model.predict(n_periods=len(test))
    mse = mean_squared_error(test, predictions)
    print("Mean Squared Error: {:.4f}".format(mse))

    # plotting the points
    plt.plot([idx for idx in range(len(test))], test)
    plt.plot([idx for idx in range(len(predictions))], predictions, color="green")

    # function to show the plot
    plt.show()


def train_and_predict(history, prediction_batch):
    model = ARIMA(history, order=(30, 0, 0))
    model_fit = model.fit()
    y_pred = model_fit.predict(start=len(history), end=len(history) + prediction_batch - 1)
    return y_pred


def train_batches(first_half_csv_path: str) -> None:
    # Загрузка данных из CSV-файла
    data = pd.read_csv(first_half_csv_path, parse_dates=["Time"])
    data = data.set_index("Time")

    # Используем столбец Close как целевую переменную для предсказания
    target = data["Close"]

    # Разделение данных на обучающую и тестовую выборки
    train_size = int(len(target) * 0.9)
    start_offset = int(len(target) * 0.7)
    train, test = target[start_offset:train_size], target[train_size:]


    history = [x for x in train]
    prediction_batch = 2000
    N_test_observations = len(test) // prediction_batch
    model_predictions = []

    futures = []
    with ProcessPoolExecutor(max_workers=20) as executor:
        for time_point in tqdm(range(N_test_observations)):
            predict_future = executor.submit(train_and_predict, copy(history), prediction_batch)
            futures.append(predict_future)

            history.extend(test[time_point * prediction_batch:time_point * prediction_batch + prediction_batch])

        for time_point in tqdm(range(N_test_observations)):
            result = futures[time_point].result()
            model_predictions.extend(result)

    # Вычисление среднеквадратической ошибки
    # mse = mean_squared_error(test, model_predictions)
    # print("Mean Squared Error: {:.4f}".format(mse))

    # plotting the points
    plt.plot([idx for idx in range(len(model_predictions))], test[:len(model_predictions)])
    plt.plot([idx for idx in range(len(model_predictions))], model_predictions, color="green")

    # function to show the plot
    plt.show()


def cache_second_half(model: MLModel, second_half_csv_path: str) -> None:
    pass
