from typing import Tuple

from pyalgotrade import plotter, strategy
from pyalgotrade.stratanalyzer import drawdown, returns, sharpe, trades
from pyalgotrade.utils import stats
from pydantic import BaseModel


class TesterResult(BaseModel):
    final_portfolio: float
    annual_return: float
    total_trades: int
    profitable_trades: int
    unprofitable_trades: int


class StrategyTester:
    def __init__(self, strategy: strategy.BacktestingStrategy):
        self._strategy = strategy
        self._setup_analyzers()

    def _setup_analyzers(self):
        self._returns_analyzer = returns.Returns()
        self._strategy.attachAnalyzer(self._returns_analyzer)

        self._sharpe_ratio_analyzer = sharpe.SharpeRatio()
        self._strategy.attachAnalyzer(self._sharpe_ratio_analyzer)

        self._draw_down_analyzer = drawdown.DrawDown()
        self._strategy.attachAnalyzer(self._draw_down_analyzer)

        self._trades_analyzer = trades.Trades()
        self._strategy.attachAnalyzer(self._trades_analyzer)

    def run(self, plot: bool) -> TesterResult:
        plt = None
        if plot:
            plt = plotter.StrategyPlotter(self._strategy, True, plotBuySell=True, True)

        self._strategy.run()

        self._print_total_results()
        self._print_trades_result()

        if plt:
            plt.plot()

        return TesterResult(
            final_portfolio=self._strategy.getResult(),
            annual_return=self._returns_analyzer.getCumulativeReturns()[-1] * 100,
            total_trades=self._trades_analyzer.getCount(),
            profitable_trades=self._trades_analyzer.getProfitableCount(),
            unprofitable_trades=self._trades_analyzer.getUnprofitableCount(),

        )

    def _print_total_results(self):

        print("Final portfolio value: $%.2f" % self._strategy.getResult())
        print("Anual return: %.2f %%" % (self._returns_analyzer.getCumulativeReturns()[-1] * 100))
        print("Average daily return: %.2f %%" % (
                stats.mean(self._returns_analyzer.getReturns()) * 100))
        print("Std. dev. daily return: %.4f" % (stats.stddev(self._returns_analyzer.getReturns())))
        print("Sharpe ratio: %.2f" % (self._sharpe_ratio_analyzer.getSharpeRatio(0)))

    def _print_trades_result(self):
        print('Total trades: %d' % self._trades_analyzer.getCount())
        if self._trades_analyzer.getCount() > 0:
            profits = self._trades_analyzer.getAll()
            print('Avg. profit: $%2.f' % (profits.mean()))
            print('Profits std. dev.: $%2.f' % (profits.std()))
            print('Max. profit: $%2.f' % (profits.max()))
            print('Min. profit: $%2.f' % (profits.min()))
            all_returns = self._trades_analyzer.getAllReturns()
            print('Avg. return: %2.f %%' % (all_returns.mean() * 100))
            print('Returns std. dev.: %2.f %%' % (all_returns.std() * 100))
            print('Max. return: %2.f %%' % (all_returns.max() * 100))
            print('Min. return: %2.f %%' % (all_returns.min() * 100))

        print('')
        print('Profitable trades: %d' % (self._trades_analyzer.getProfitableCount()))
        if self._trades_analyzer.getProfitableCount() > 0:
            profits = self._trades_analyzer.getProfits()
            print('Avg. profit: $%2.f' % (profits.mean()))
            print('Profits std. dev.: $%2.f' % (profits.std()))
            print('Max. profit: $%2.f' % (profits.max()))
            print('Min. profit: $%2.f' % (profits.min()))
            pos_returns = self._trades_analyzer.getPositiveReturns()
            print('Avg. return: %2.f %%' % (pos_returns.mean() * 100))
            print('Returns std. dev.: %2.f %%' % (pos_returns.std() * 100))
            print('Max. return: %2.f %%' % (pos_returns.max() * 100))
            print('Min. return: %2.f %%' % (pos_returns.min() * 100))

        print('')
        print('Unprofitable trades: %d' % (self._trades_analyzer.getUnprofitableCount()))
        if self._trades_analyzer.getUnprofitableCount() > 0:
            losses = self._trades_analyzer.getLosses()
            print('Avg. loss: $%2.f' % (losses.mean()))
            print('Losses std. dev.: $%2.f' % (losses.std()))
            print('Max. loss: $%2.f' % (losses.min()))
            print('Min. loss: $%2.f' % (losses.max()))
            neg_returns = self._trades_analyzer.getNegativeReturns()
            print('Avg. return: %2.f %%' % (neg_returns.mean() * 100))
            print('Returns std. dev.: %2.f %%' % (neg_returns.std() * 100))
            print('Max. return: %2.f %%' % (neg_returns.max() * 100))
            print('Min. return: %2.f %%' % (neg_returns.min() * 100))
