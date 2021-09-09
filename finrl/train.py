from datetime import datetime
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
    elif config_suffix == '7':
        from finrl.apps.crypto_etc import crypto_config7 as crypto_config
    elif config_suffix == '8':
        from finrl.apps.crypto_etc import crypto_config8 as crypto_config
    elif config_suffix == '9':
        from finrl.apps.crypto_etc import crypto_config9 as crypto_config
    elif config_suffix == '10':
        from finrl.apps.crypto_etc import crypto_config10 as crypto_config
    elif config_suffix == '11':
        from finrl.apps.crypto_etc import crypto_config11 as crypto_config
    elif config_suffix == '12':
        from finrl.apps.crypto_etc import crypto_config12 as crypto_config
    else:
        raise 'config_suffix is not in define'
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = BinanceDownloader(
        start_date=crypto_config.ENUMERATE_START_DATE,
        end_date=crypto_config.ENUMERATE_END_DATE,
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


    enumerate_date_paras = []
    for idx, END_DATE in enumerate(
        pd.date_range(
            start=crypto_config.ENUMERATE_START_TRADE_DATE,
            end=crypto_config.ENUMERATE_END_DATE,
            freq='M') + pd.DateOffset(days=1)
        ):
        enumerate_date_paras.append([
        (datetime.strptime(crypto_config.ENUMERATE_START_DATE, '%Y-%m-%d') + pd.DateOffset(months=idx)).strftime('%Y-%m-%d'),
        (datetime.strptime(crypto_config.ENUMERATE_START_TRADE_DATE, '%Y-%m-%d') + pd.DateOffset(months=idx)).strftime('%Y-%m-%d'),
        END_DATE.strftime('%Y-%m-%d')])

    # init
    model = None
    for enumerate_date_para in enumerate_date_paras:
        START_DATE, START_TRADE_DATE, END_DATE = enumerate_date_para
        # Training & Trading data split
        train = data_split(processed_full, START_DATE, START_TRADE_DATE)
        trade = data_split(processed_full, START_TRADE_DATE, END_DATE)

        stock_dimension = len(train.tic.unique())
        state_space = (
            1
            + 2 * stock_dimension
            + len(crypto_config.TECHNICAL_INDICATORS_LIST) * stock_dimension
        )

        env_kwargs = {
            "hmax": 100*500,
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

        for model_type in ['a2c', 'ddpg', 'td3', 'sac']:
            print(f"=============={model_type}===========")
            if model is None:
                model = agent.get_model(model_type)
            else:
                from stable_baselines3 import A2C, TD3, SAC, DDPG
                if model_type == 'a2c':
                    model = A2C.load(f'./results/{crypto_config.RESULTS_DIR}/model_a2c_{LAST_TRADE_DATE}')
                if model_type == 'td3':
                    model = TD3.load(f'./results/{crypto_config.RESULTS_DIR}/model_td3_{LAST_TRADE_DATE}')
                if model_type == 'sac':
                    model = SAC.load(f'./results/{crypto_config.RESULTS_DIR}/model_sac_{LAST_TRADE_DATE}')
                if model_type == 'ddpg':
                    model = DDPG.load(f'./results/{crypto_config.RESULTS_DIR}/model_ddpg_{LAST_TRADE_DATE}')

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
                f"./results/{crypto_config.RESULTS_DIR}/df_account_value_{model_type}_{START_TRADE_DATE}.csv"
            )
            df_actions.to_csv(f'./results/{crypto_config.RESULTS_DIR}/df_actions_{model_type}_{START_TRADE_DATE}.csv')
            print("./results/" + crypto_config.RESULTS_DIR + "/df_actions_" + model_type + ".csv")

            crypto_backtest_plot(
                account_value=df_account_value,
                baseline_tickers=["BTC/USDT", "ETH/USDT", "LTC/USDT", "XLM/USDT", "BNB/USDT"],
                baseline_start=START_DATE,
                baseline_end=END_DATE,
                pngname=f'{model_type}_returns_{START_TRADE_DATE}',
                config_suffix=config_suffix
            )
            trained.save(f'./results/{crypto_config.RESULTS_DIR}/model_{model_type}_{START_TRADE_DATE}')
        LAST_TRADE_DATE = START_TRADE_DATE.copy()



# from stable_baselines3 import DDPG
# trained = DDPG.load('/Users/bacon_huang/Downloads/model_ddpg')

# config_suffix = '1'
# if config_suffix == '1':
#     from finrl.apps.crypto_etc import crypto_config1 as crypto_config
# elif config_suffix == '2':
#     from finrl.apps.crypto_etc import crypto_config2 as crypto_config
# elif config_suffix == '3':
#     from finrl.apps.crypto_etc import crypto_config3 as crypto_config
# elif config_suffix == '4':
#     from finrl.apps.crypto_etc import crypto_config4 as crypto_config
# elif config_suffix == '5':
#     from finrl.apps.crypto_etc import crypto_config5 as crypto_config
# elif config_suffix == '6':
#     from finrl.apps.crypto_etc import crypto_config6 as crypto_config
# elif config_suffix == '7':
#     from finrl.apps.crypto_etc import crypto_config7 as crypto_config
# elif config_suffix == '8':
#     from finrl.apps.crypto_etc import crypto_config8 as crypto_config
# elif config_suffix == '9':
#     from finrl.apps.crypto_etc import crypto_config9 as crypto_config
# elif config_suffix == '10':
#     from finrl.apps.crypto_etc import crypto_config10 as crypto_config
# elif config_suffix == '11':
#     from finrl.apps.crypto_etc import crypto_config11 as crypto_config
# elif config_suffix == '12':
#     from finrl.apps.crypto_etc import crypto_config12 as crypto_config
# else:
#     raise 'config_suffix is not in define'

# print("==============Start Fetching Data===========")
# df = BinanceDownloader(
#     start_date=crypto_config.START_DATE,
#     end_date=crypto_config.END_DATE,
#     ticker_list=crypto_config.DOW_30_TICKER,
# ).fetch_data()
# crypto_config.DOW_30_TICKER = list(df.tic.value_counts().nlargest(1, keep='all').index)
# if len(crypto_config.DOW_30_TICKER) < 3:
#     raise 'no tic'
# else:
#     df = df[df['tic'].isin(crypto_config.DOW_30_TICKER)]

# print("==============Start Feature Engineering===========")
# fe = FeatureEngineer(
#     use_technical_indicator=True,
#     tech_indicator_list=crypto_config.TECHNICAL_INDICATORS_LIST,
#     use_turbulence=True,
#     user_defined_feature=False,
# )
# processed = fe.preprocess_data(df)

# list_ticker = processed["tic"].unique().tolist()
# list_date = list(pd.date_range(processed['date'].min(), processed['date'].max(), freq='30min').astype(str))
# combination = list(itertools.product(list_date, list_ticker))

# processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
# processed_full = processed_full[processed_full['date'].isin(processed['date'])]
# processed_full = processed_full.sort_values(['date', 'tic'])

# processed_full = processed_full.fillna(0)

# # Training & Trading data split
# train = data_split(processed_full, crypto_config.START_DATE, crypto_config.START_TRADE_DATE)
# trade = data_split(processed_full, crypto_config.START_TRADE_DATE, crypto_config.END_DATE)

# stock_dimension = len(train.tic.unique())
# state_space = (
#     1
#     + 2 * stock_dimension
#     + len(crypto_config.TECHNICAL_INDICATORS_LIST) * stock_dimension
# )

# env_kwargs = {
#     "hmax": 100,
#     "initial_amount": 1000000,
#     "buy_cost_pct": 0.001,
#     "sell_cost_pct": 0.001,
#     "state_space": state_space,
#     "stock_dim": stock_dimension,
#     "tech_indicator_list": crypto_config.TECHNICAL_INDICATORS_LIST,
#     "action_space": stock_dimension,
#     "reward_scaling": 1e-4
# }


# print("==============Start Trading===========")
# turbulence_threshold = int(np.quantile(train.turbulence, 0.99))
# e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=turbulence_threshold, **env_kwargs)

# df_account_value, df_actions = DRLAgent.DRL_prediction(
#     model=trained, environment=e_trade_gym
# )
