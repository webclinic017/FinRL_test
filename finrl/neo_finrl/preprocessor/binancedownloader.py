"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import ccxt


class BinanceDownloader:
    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def retry_fetch_ohlcv(self, exchange, max_retries, symbol, timeframe, since, limit):
        num_retries = 0
        try:
            num_retries += 1
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
            return ohlcv
        except Exception:
            if num_retries > max_retries:
                raise Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

    def scrape_ohlcv(self, exchange, max_retries, symbol, timeframe, since, limit):
        earliest_timestamp = exchange.parse8601(self.end_date + '00:00:00Z')
        # earliest_timestamp = exchange.milliseconds()
        timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
        timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
        timedelta = limit * timeframe_duration_in_ms
        all_ohlcv = []
        while True:
            # timeframe=30m ==> timedelta: 1800*1000 毫秒是一筆，共1000筆 ==> 抓 1800*1000*1000時間內的資料
            fetch_since = earliest_timestamp - timedelta
            ohlcv = self.retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
            # if we have reached the beginning of history

            if ohlcv is None:
                break
            elif len(ohlcv) == 0:
                break
            elif ohlcv[0][0] >= earliest_timestamp:
                break

            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
            # if we have reached the checkpoint
            if fetch_since < since:
                break
        return all_ohlcv

    def fetch_data(self, timeframe='30m') -> pd.DataFrame:
        exchange = getattr(ccxt, 'binance')({
          'enableRateLimit': True,  # required by the Manual
        })
        # convert since from string to milliseconds integer if needed
        print(self.start_date)
        if isinstance(self.start_date, str):
            start_date = exchange.parse8601(self.start_date + '00:00:00Z')
        exchange.load_markets()
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            ohlcv = self.scrape_ohlcv(exchange, 3, tic, timeframe, start_date, 1000)
            ohlcv_df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            ohlcv_df['tic'] = tic
            ohlcv_df['day'] = (pd.to_datetime(ohlcv_df.date, unit='ms') + pd.Timedelta('08:00:00')).dt.dayofweek
            ohlcv_df['date'] = (pd.to_datetime(ohlcv_df.date, unit='ms') + pd.Timedelta('08:00:00')).dt.strftime('%Y-%m-%d %H:%M:%S')
            data_df = data_df.append(ohlcv_df)
        data_df = data_df[(self.start_date <= data_df.date) & (data_df.date <= self.end_date)]
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)

        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        return data_df



