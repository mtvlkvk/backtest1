import datetime
import numpy as np
import pandas as pd
import sklearn
# from pandas.io.data import DataReader
import pandas_datareader.data as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
# from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from all_imports import yf


def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    В общем, сильно поменял. Результат тот же, оригинал в книжке
    """
    # Obtain stock information from Yahoo Finance
    '''
    # так было, но переписал с использованием yf, который мне понятнее. Плюс не понимаю,
    # почему конкретно здеса автор использует DataReader
    ts = web.DataReader(
        symbol, "yahoo",
        start_date - datetime.timedelta(days=365),
        end_date)
    '''
    ts = yf.download(symbol, start=start_date - datetime.timedelta(days=365), end=end_date)
    tsret = pd.DataFrame({'Today': ts['Adj Close'].pct_change() * 100, 'Volume': ts['Volume']})
    for i in range(0, lags):
        tsret[f'Lag{str(i+1)}'] = ts["Adj Close"].shift(i + 1).pct_change() * 100.0
    tsret.loc[abs(tsret['Today']) < 0.0001, 'Today'] = 0.0001
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    return tsret

