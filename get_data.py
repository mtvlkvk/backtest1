import pandas as pd
import yfinance as yf

def get_yahoo_data(yahoo_id, start_date, end_date):
    csv_path = './csv/{}.csv'.format(yahoo_id)
    try:
        df = pd.read_csv(csv_path)
        print('Loaded {} from cache'.format(yahoo_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Yahoo'.format(yahoo_id))
        df = yf.download(yahoo_id, start=start_date, end=end_date)
        # df.index = pd.to_datetime(df.index)
        df.to_csv(csv_path)
        print('Cached {} at {}'.format(yahoo_id, csv_path))
    return df