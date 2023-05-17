import json
from pathlib import Path

import numpy
import pandas
import pyalgotrade.dataseries
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class MLModel:
    def __init__(self, model: Sequential, scaler: MinMaxScaler, required_prices: int, predicts_forward: int):
        self._model = model
        self._scaler = scaler
        self._required_prices = required_prices
        self._cache = {}
        self._predicts_forward = predicts_forward

    @property
    def loop_back(self) -> int:
        return self._required_prices

    @property
    def predicts_forward(self) -> int:
        return self._predicts_forward

    def ready(self, price_series: pyalgotrade.dataseries.DataSeries) -> bool:
        older_prices = price_series[-self._required_prices:]
        return len(older_prices) == self._required_prices

    def load_cache(self, file_path: Path) -> None:
        with open(Path(file_path)) as file:
            data = json.load(file)

        for obj in data:
            x_values = numpy.asarray(obj["x_values"], dtype=float)
            self._cache[tuple(x_values)] = float(obj["y"])

    def save_cache(self, file_path: Path):
        data = []
        for x_values in self._cache:
            data.append({
                "x_values": [str(value) for value in x_values],
                "y": str(self._cache[x_values]),
            })

        with open(Path(file_path), "w") as file:
            json.dump(data, file, indent=4)

    def create_cache(self, X_train: numpy.ndarray, Y_train: numpy.ndarray, X: numpy.ndarray, Y: numpy.ndarray) -> None:
        # self._create_cache_with_N_bach(X, Y, batches=4)
        self._create_cache_with_N_bach(X_train, Y_train, X, Y, batches=5)

    def _create_cache_with_N_bach(self, X_train: numpy.ndarray, Y_train: numpy.ndarray, X: numpy.ndarray, Y: numpy.ndarray, batches: int) -> None:

        x_len = len(X)
        predict_x_len = x_len // batches
        if batches * predict_x_len < x_len:
            predict_x_len += x_len - batches * predict_x_len

        for batch_idx in range(batches):
            start_idx = batch_idx * predict_x_len
            end_index = min(predict_x_len * (batch_idx + 1), x_len)

            print(f"Predict from {start_idx} to {end_index}")
            first_half_y_pred = self._model.predict(X[start_idx:end_index])

            for idx in range(start_idx, end_index):
                self._cache[tuple(X[idx].flatten())] = first_half_y_pred[idx - start_idx][0]

            if batch_idx < batches - 1:
                early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
                self._model.fit(
                    numpy.vstack((X_train, X[:end_index])), numpy.hstack((Y_train, Y[:end_index])),
                    validation_data=(X[end_index:], Y[end_index:]), epochs=4,
                    batch_size=4000, verbose=1, callbacks=[early_stop],
                )

    def _create_cache_with_2_bach(self, X: numpy.ndarray, Y: numpy.ndarray):
        predict_x_len = len(X) // 2

        first_half_y_pred = self._model.predict(X[:predict_x_len])
        for idx in range(predict_x_len):
            self._cache[tuple(X[idx].flatten())] = first_half_y_pred[idx][0]

        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
        self._model.fit(X[:predict_x_len], Y[:predict_x_len],
                        validation_data=(X[predict_x_len:], Y[predict_x_len:]), epochs=2,
                        batch_size=4000, verbose=1, callbacks=[early_stop])

        second_half_y_pred = self._model.predict(X[len(X) // 2:])

        for idx in range(predict_x_len):
            real_idx = idx + predict_x_len
            self._cache[tuple(X[real_idx].flatten())] = second_half_y_pred[idx][0]

    def predict(self, price_series: pyalgotrade.dataseries.DataSeries) -> float:
        last_prices = numpy.array(price_series[-self.loop_back:])
        last_values_scaled = self._scaler.transform(last_prices.reshape(-1, 1))

        if tuple(last_values_scaled.flatten()) in self._cache:
            next_value_scaled_raw = self._cache[tuple(last_values_scaled.flatten())]
            next_value_scaled = numpy.array([[next_value_scaled_raw]])
        else:
            next_value_scaled = self._model.predict(
                last_values_scaled.reshape(1, self.loop_back, 1))

        # Преобразование предсказанного значения обратно в исходный масштаб
        next_value = self._scaler.inverse_transform(next_value_scaled)[0][0]
        return next_value
