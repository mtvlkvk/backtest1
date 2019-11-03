import datetime
import os, os.path
# import pandas as pd
# from sklearn.qda import QDA
from all_imports import pd, np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import LinearSVC, SVC

from strategy import Strategy
from event import SignalEvent
from backtest import Backtest
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from create_lagged_series import *
from get_data import get_polo_data

from get_data import get_yahoo_data

class SPYDailyForecastStrategy(Strategy):
    """
    S&P500 forecast strategy. It uses a Quadratic Discriminant Analyser to predict the returns
     for a subsequent time period and then generated long/exit signals based on the prediction."""
    def __init__(self, bars, events, start_date):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.start_date = start_date
        self.datetime_now = datetime.datetime.utcnow()
        self.model_start_date = datetime.datetime(2001,1,10)
        # TODO Понятно, что эту хрень нужно менять. Вот только откуда брать даты ...
        self.model_end_date = datetime.datetime(2005,12,31)
        self.model_start_test_date = datetime.datetime(2005,1,1)
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model = self.create_symbol_forecast_model()

    def create_symbol_forecast_model(self):
        snpret = create_lagged_polo(
            self.symbol_list[0], self.start_date, lags=5
        )
        # Use the prior two days of returns as predictor # values, with direction as the response
        X = snpret[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"]]
        y = snpret["Direction"]
        # Create training and test sets
        '''
        start_test = self.model_start_test_date
        X_train = X[X.index < start_test]
        X_test = X[X.index >= start_test]
        y_train = y[y.index < start_test]
        y_test = y[y.index >= start_test]
        '''

        model = LogisticRegression()
        # ниже все возможные модели, куда необходимо также добавить Keras
        '''
        model = LDA()
        model = QDA()
        model = LinearSVC()
        model = SVC(
            C=1000000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
            max_iter=-1, probability=False, random_state=None,
            shrinking=True, tol=0.001, verbose=False
        )
        model = RandomForestClassifier(
            n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False,
            n_jobs=1, random_state=None, verbose=0
        )
        '''
        model.fit(X, y)
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
                self.symbol_list[0], 'returns', N=5
                # TODO: на самом деле не ясно, что здесь он хотел использовать returns или adj_close
            )
            pred_series = pd.Series(
                {
                    'Lag1': lags[0] * 100.0,
                    'Lag2': lags[1] * 100.0,
                    'Lag3': lags[2] * 100.0,
                    'Lag4': lags[3] * 100.0,
                    'Lag5': lags[4] * 100.0
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
    symbol_list = ['USDT_BTC']
    # get_polo_data(symbol_list)
    csv_dir = './csv'
    symbol_data = {}
    initial_capital = 100000.0
    heartbeat = 0.0
    start_date = datetime.datetime(2018, 1, 1, 0, 0, 0)
    # start_date_sec = start_date.timestamp()
    # today_sec = datetime.datetime.now().timestamp()
    delta_sec = datetime.datetime.now().timestamp() - start_date.timestamp()

    for s in symbol_list:
        symbol_data[s] = get_polo_data(pair=s, frame=delta_sec)
        symbol_data[s].columns = ['open', 'high',
                                  'low', 'close', 'volume', 'adj_close']
        # start_date = symbol_data[s].index[0]

    backtest = Backtest(
        csv_dir, symbol_list, symbol_data, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, SPYDailyForecastStrategy
    )
    backtest.simulate_trading()