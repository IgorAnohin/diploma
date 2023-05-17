from typing import List

import pyalgotrade
from pyalgotrade import strategy
from pyalgotrade.barfeed.csvfeed import BarFeed

from diploma.models.base import MLModel


class MLStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed: BarFeed, figi: str, model: MLModel):
        super(MLStrategy, self).__init__(feed)
        self._figi = figi
        self.__prices = feed[figi].getPriceDataSeries()

        self._model = model

        self._position = None
        self._enter_price = None

    def onEnterCanceled(self, position: pyalgotrade.strategy.position.Position):
        self._position = None

    def onExitOk(self, position: pyalgotrade.strategy.position.Position):
        self._position = None

    def onExitCanceled(self, position: pyalgotrade.strategy.position.Position):
        self._position.exitMarket()

    def predicted_price_enough(self) -> bool:
        last_price = self.__prices[-1]

        if not self._model.ready(self.__prices[:-self._model.predicts_forward]):
            return False

        next_values = [self._model.predict(self.__prices[:-value]) for value in
                       range(self._model.predicts_forward, 0, -1)]
        next_value = self._model.predict(self.__prices)
        next_values.append(next_value)

        # # print(f"next_value={next_value}, last_price={last_price}, {next_value / last_price}%")
        # # return next_value / last_price > 1.02
        # # greater_than_current_price_count = sum(1 for next_value in next_values if next_value
        # > last_price)
        # # print(f"Current value: {last_price}, Next values: {next_values}. Coeff {
        # greater_than_current_price_count / len(next_values)}")
        # # return greater_than_current_price_count / len(next_values) > 0.7
        # growing_count = 0
        # for prev, next in zip(next_values[:-1], next_values[1:]):
        #     if next > prev:
        #         growing_count += 1

        return (
            # self._has_growing_trand(next_values) and
            self._weighted_average_bigger_last(last_price, next_values) and
            # self._next_value_bigger_than_trashhold(last_price, next_values) and
            # self.predictions_close_to_last_price(last_price, next_values) and
            True
        )

    def predictions_close_to_last_price(self, last_price: float, next_values: List[float]) -> bool:
        weighted_average = sum(next_values) / len(next_values)
        if weighted_average / last_price > 1.2:
            print(
                f"Skip because average forecasted {weighted_average} and last price is "
                f"{last_price}. Ratio: {weighted_average / last_price}")
            return False
        return True

    def _weighted_average_bigger_last(self, last_price: float, next_values: List[float]) -> bool:
        target_price = last_price * 1.02  # Рост на 2%
        weighted_average = sum(next_values) / len(next_values)

        return weighted_average > target_price

    def _next_value_bigger_than_trashhold(self, last_price: float,
                                          next_values: List[float]) -> bool:
        greater_than_current_price_count = sum(
            1 for next_value in next_values if next_value > last_price)
        return greater_than_current_price_count / len(next_values) > 0.7

    @staticmethod
    def _has_growing_trand(next_values: List[float]) -> bool:
        growing_count = 0
        for prev, next in zip(next_values[:-1], next_values[1:]):
            if next > prev:
                growing_count += 1

        return growing_count / len(next_values) > 0.7

    def onBars(self, bars) -> None:
        if not self._position:
            if self.predicted_price_enough():
                shares = self.getBroker().getCash() * 0.9 // bars[self._figi].getPrice()
                self._enter_price = bars[self._figi].getPrice()
                self._position = self.enterLong(self._figi, shares, True)
        elif not self._position.exitActive():
            price_diff = self._enter_price - bars[self._figi].getPrice()
            price_changed_on_2_percents = abs(price_diff) / self._enter_price >= 0.02

            if price_changed_on_2_percents:
                self._enter_price = None
                self._position.exitMarket()
