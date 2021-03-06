import pandas as pd
import numpy as np

from pyfolio import timeseries
import pyfolio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from copy import deepcopy

from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.apps import config, crypto_config


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret= df.copy()
    strategy_ret['date'] = pd.to_datetime(strategy_ret['date'])
    strategy_ret.set_index('date', drop = False, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts

def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def crypto_backtest_plot(
    account_value,
    baseline_start,
    baseline_end,
    baseline_tickers,
    value_col_name="account_value",
    pngname='model_returns',
    config_suffix=''
):

    df = deepcopy(account_value)
    df = df[(baseline_start <= df.date) & (df.date <= baseline_end)]
    test_returns = get_daily_return(df, value_col_name=value_col_name)
    test_returns = (test_returns+1).cumprod()

    from finrl.neo_finrl.preprocessor.binancedownloader import BinanceDownloader
    for baseline_ticker in baseline_tickers:
        baseline_df = BinanceDownloader(
            start_date=baseline_start,
            end_date=baseline_end,
            ticker_list=[baseline_ticker],
        ).fetch_data()

        baseline_returns = get_daily_return(baseline_df, value_col_name="close")
        baseline_returns = (baseline_returns+1).cumprod()

        merge_baseline_test = pd.concat([baseline_returns, test_returns], axis=1)
        merge_baseline_test.columns = ['benchmark', 'backtest']
        plt.figure(figsize=(12, 5))
        plt.xlabel('Number of requests every 10 minutes')

        merge_baseline_test.benchmark.plot(color='blue', grid=True, label='benchmark')
        merge_baseline_test.backtest.plot(color='red', grid=True, label='backtest')

        plt.legend(loc=2)
        plt.ylim(0, 2)
        plt.show()
        plt.savefig(f"./results/results_{config_suffix}/{pngname}__{baseline_ticker.replace('/', '')}.png")


def backtest_plot(
    account_value,
    baseline_start=config.START_TRADE_DATE,
    baseline_end=config.END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):

    df = deepcopy(account_value)
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )


def get_baseline(ticker, start, end):
    dji = YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()
    return dji


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions['transactions'].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions['date']
    df_trx.index.name = ''

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: True if x > 0 else False)
        selling_signal = df_trx_temp_sign.apply(lambda x: True if x < 0 else False)

        tic_plot = df_trade[(df_trade['tic'] == df_trx_temp.name) & (df_trade['date'].isin(df_trx.index))]['close']
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color='g', lw=2.)
        plt.plot(tic_plot, '^', markersize=10, color='m', label='buying signal', markevery=buying_signal)
        plt.plot(tic_plot, 'v', markersize=10, color='k', label='selling signal', markevery=selling_signal)
        plt.title(f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal==True]) + len(selling_signal[selling_signal==True])}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha='right')
        plt.show()
