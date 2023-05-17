from pyalgotrade.barfeed.csvfeed import BarFeed
from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
from pyalgotrade import strategy


class SMACrossOverStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed: BarFeed, figi: str, sma_period: int):
        super(SMACrossOverStrategy, self).__init__(feed)
        self._figi = figi
        self._position = None

        self._prices = feed[figi].getPriceDataSeries()
        self._sma = ma.SMA(self._prices, sma_period)

    def onEnterCanceled(self, position):
        self._position = None

    def onExitOk(self, position):
        self._position = None

    def onExitCanceled(self, position):
        self._position.exitMarket()

    def onBars(self, bars):
        if not self._position:
            if cross.cross_above(self._prices, self._sma) > 0:
                shares = int(self.getBroker().getCash() * 0.9 / bars[self._figi].getPrice())
                self._position = self.enterLong(self._figi, shares, True)
        elif self._position:
            if not self._position.exitActive() and cross.cross_below(self._prices, self._sma) > 0:
                self._position.exitMarket()

