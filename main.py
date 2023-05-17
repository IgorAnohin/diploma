import csv
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from pyalgotrade.barfeed import quandlfeed

from diploma.models import arima, catb, lstm
from diploma.strategies.ml_strategy import MLStrategy
from diploma.strategies.sma_cross_over import SMACrossOverStrategy
from diploma.strategy_tester import StrategyTester, TesterResult


def create_halfs_csv(csv_path: Path) -> Tuple[str, str]:
    # Open the input CSV file
    second_half_path = f"tmp/second_half_{csv_path.name}"
    first_half_path = f"tmp/first_half_{csv_path.name}"
    with open(csv_path) as input_file:
        # Create output files for two parts
        with open(first_half_path, 'w', newline='') as first_half_file:
            with open(second_half_path, 'w', newline='') as second_half_file:
                # Initialize CSV readers and writers
                reader = csv.reader(input_file)
                first_half_writer = csv.writer(first_half_file)
                second_half_writer = csv.writer(second_half_file)

                # Write headers to both output files
                headers = next(reader)
                first_half_writer.writerow(headers)
                second_half_writer.writerow(headers)

                # Write rows to output files based on a condition
                rows = [row for row in reader]
                for first_half in rows[:len(rows) // 2]:
                    first_half_writer.writerow(first_half)

                for second_half in rows[len(rows) // 2:]:
                    second_half_writer.writerow(second_half)

    return first_half_path, second_half_path

def perform(csv_path: Path, plot: bool) -> Tuple[float, TesterResult]:
    instrument = csv_path.name.split("_")[0]

    print(f"Working with {instrument}")

    first_half_path, second_half_path = create_halfs_csv(csv_path)

    # ARIMA
    # ml_model = arima.train(csv_path)

    # CatBoost
    # ml_model = catb.train(csv_path)
    # ml_model = catb.train_returns(csv_path)
    # ml_model = catb.train_with_batches(csv_path)

    # LSTM
    # ml_model = lstm.train(csv_path)
    # ml_model = lstm.train_with_returns(csv_path)
    ml_model = lstm.train_complex(csv_path)

    lstm.cache_second_half(ml_model, csv_path)

    feed = quandlfeed.Feed()
    feed.setColumnName("datetime", "Time")
    feed.setDateTimeFormat('%Y-%m-%d %H:%M:%S+00:00')
    feed.addBarsFromCSV(instrument, second_half_path)

    strategy = MLStrategy(feed, instrument, ml_model)
    # strategy = SMACrossOverStrategy(feed, instrument, 18)

    data = pd.read_csv(second_half_path)
    close_prices = data['Close']
    buy_and_hold = close_prices[len(close_prices)-1] / close_prices[0]

    tester = StrategyTester(strategy)
    return buy_and_hold, tester.run(plot)


def main(plot):
    results = {}

    start_time = time.time()

    data_dir = Path("data")
    for csv_path in data_dir.iterdir():
        results[csv_path] = perform(csv_path, plot)

    # csv_path = Path("data/TCSG_5min__72594_2021-05-04-21-00_2023-05-04-21-00.csv")
    # csv_path = Path("data/SBER_5min__82695_2021-05-04-21-00_2023-05-04-21-00.csv")
    # csv_path = Path("data/NVTK_5min__81808_2021-05-04-21-00_2023-05-04-21-00.csv")
    # results[csv_path] = perform(csv_path, plot)

    print("Results:")
    for csv_path in results:
        buy_and_hold, result = results[csv_path]
        print(csv_path.name.split("_")[0], buy_and_hold, result.annual_return, result.total_trades, result.profitable_trades, result.unprofitable_trades)

    end_time = time.time()
    print("Took", end_time - start_time)


if __name__ == "__main__":
    main(True)
