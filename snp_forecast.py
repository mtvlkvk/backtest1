#!/usr/bin/python
# -*- coding: utf-8 -*-
# snp_forecast.py
from __future__ import print_function
import datetime
import os, os.path
# import pandas as pd
# from sklearn.qda import QDA
from all_imports import pd, np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from create_lagged_series import create_lagged_series

from get_data import get_yahoo_data

class SPYDailyForecastStrategy(Strategy):
    """
    S&P500 forecast strategy. It uses a Quadratic Discriminant Analyser to predict the returns
     for a subsequent time period and then generated long/exit signals based on the prediction."""
    def __init__(self, bars, events, start_date):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.datetime_now = datetime.datetime.utcnow()
        self.model_start_date = datetime.datetime(2001,1,10)
        self.model_end_date = datetime.datetime(2005,12,31)
        self.model_start_test_date = datetime.datetime(2005,1,1)
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model = self.create_symbol_forecast_model()

    def create_symbol_forecast_model(self): # Create a lagged series of the S&P500 US stock market index
        snpret = create_lagged_series(
            self.symbol_list[0], self.model_start_date, self.model_end_date, lags = 5
        )
        # Use the prior two days of returns as predictor # values, with direction as the response
        X = snpret[["Lag1", "Lag2"]]
        y = snpret["Direction"]
        # Create training and test sets
        start_test = self.model_start_test_date
        X_train = X[X.index < start_test]
        X_test = X[X.index >= start_test]
        y_train = y[y.index < start_test]
        y_test = y[y.index >= start_test]
        model = QDA()
        model.fit(X_train, y_train)
        return model

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents based on market data.
        """
        sym = self.symbol_list[0]
        dt = self.datetime_now
        if event.type == 'MARKET':
            self.bar_index += 1
        if self.bar_index > 5:
            lags = self.bars.get_latest_bars_values(
                # self.symbol_list[0], "returns", N=3
                self.symbol_list[0], 'returns', N=3
                # TODO: на самом деле не ясно, что здесь он хотел использовать returns или adj_close
            )
            pred_series = pd.Series(
                {
                    'Lag1': lags[1] * 100.0,
                    'Lag2': lags[2] * 100.0
                }
            )
            pred = self.model.predict(np.array(pred_series).reshape(1, -1))
            if pred > 0 and not self.long_market:
                self.long_market = True
                signal = SignalEvent(1, sym, dt, 'LONG', 1.0)
                self.events.put(signal)

            if pred < 0 and self.long_market:
                self.long_market = False
                signal = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                self.events.put(signal)

if __name__ == "__main__":
    get_yahoo_data('SPY', '2006-01-03', '2014-10-10')
    csv_dir = './csv' # CHANGE THIS!
    symbol_list = ['SPY']
    symbol_data = {}
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime.datetime(2006, 1, 3)

    for s in symbol_list:
        # Load the CSV file with no header information, indexed on date
        symbol_data[s] = pd.io.parsers.read_csv(
            os.path.join(csv_dir, "%s.csv" % s),
            header=0, index_col=0, parse_dates=True,
            names=[
                'datetime', 'open', 'high',
                'low', 'close', 'adj_close', 'volume'
                # "low", "close", "volume", "adj_close"
            ]
        ).sort_index()

    backtest = Backtest(
        csv_dir, symbol_list, symbol_data, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, SPYDailyForecastStrategy
    )
    backtest.simulate_trading()
