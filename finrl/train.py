import os
from finrl.plot import crypto_backtest_plot
import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.binancedownloader import BinanceDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import itertools


def train_stock_trading():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)

    # Training & Trading data split
    train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
        }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(
        model=model_sac, tb_log_name="sac", total_timesteps=80000
    )

    print("==============Start Trading===========")
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment = e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")


def train_crypto_trading(config_suffix):
    if config_suffix == '1':
        from finrl.apps.crypto_etc import crypto_config1 as crypto_config
    elif config_suffix == '2':
        from finrl.apps.crypto_etc import crypto_config2 as crypto_config
    elif config_suffix == '3':
        from finrl.apps.crypto_etc import crypto_config3 as crypto_config
    elif config_suffix == '4':
        from finrl.apps.crypto_etc import crypto_config4 as crypto_config
    elif config_suffix == '5':
        from finrl.apps.crypto_etc import crypto_config5 as crypto_config
    elif config_suffix == '6':
        from finrl.apps.crypto_etc import crypto_config6 as crypto_config
    else:
        raise 'config_suffix is not in define'
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = BinanceDownloader(
        start_date=crypto_config.START_DATE,
        end_date=crypto_config.END_DATE,
        ticker_list=crypto_config.DOW_30_TICKER,
    ).fetch_data()
    crypto_config.DOW_30_TICKER = list(df.tic.value_counts().nlargest(1, keep='all').index)
    if len(crypto_config.DOW_30_TICKER) < 3:
        raise 'no tic'
    else:
        df = df[df['tic'].isin(crypto_config.DOW_30_TICKER)]

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=crypto_config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max(), freq='30min').astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)

    # Training & Trading data split
    train = data_split(processed_full, crypto_config.START_DATE, crypto_config.START_TRADE_DATE)
    trade = data_split(processed_full, crypto_config.START_TRADE_DATE, crypto_config.END_DATE)

    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(crypto_config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": crypto_config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")

    for model_type in ['a2c', 'ddpg', 'ppo', 'td3']:
        print(f"=============={model_type}===========")
        model = agent.get_model(model_type)
        trained = agent.train_model(
            model=model, tb_log_name=model_type, total_timesteps=crypto_config.TOTAL_TIMESTAMPS
        )

        print("==============Start Trading===========")
        turbulence_threshold = int(np.quantile(train.turbulence, 0.99))
        e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=turbulence_threshold, **env_kwargs)

        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained, environment=e_trade_gym
        )

        # save result
        os.makedirs(f'./results/{crypto_config.RESULTS_DIR}', exist_ok=True)
        df_account_value.to_csv(
            "./results/" + crypto_config.RESULTS_DIR + "/df_account_value_" + model_type + ".csv"
        )
        df_actions.to_csv("./results/" + crypto_config.RESULTS_DIR + "/df_actions_" + model_type + ".csv")

        crypto_backtest_plot(
            df_account_value,
            baseline_ticker=["BTC/USDT", "ETH/USDT", "LTC/USDT", "XLM/USDT"],
            baseline_start=crypto_config.START_TRADE_DATE,
            baseline_end=crypto_config.END_DATE,
            pngname=f'{model_type}_returns'
        )
        trained.save(f'./results/{crypto_config.RESULTS_DIR}/model_{model_type}')
