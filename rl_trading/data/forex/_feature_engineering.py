import itertools
import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from ta.volatility import (
    AverageTrueRange,
    BollingerBands
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    PSARIndicator,
    VortexIndicator
)
from ta.momentum import (
    RSIIndicator,
    StochRSIIndicator,
    UltimateOscillator
)
from ._common import FOREX_COLS

class ForexFeEngStrategy(Enum):
    BASIC = 0 # "Financial Trading as a Game: A Deep Reinforcement Learning Approach" 
    TA = 1


def engineer_forex_features(
    data: Dict[str, pd.DataFrame],
    strategy: ForexFeEngStrategy,
    strategy_params: Dict
) -> pd.DataFrame:
    '''
    Performs feature engineering according to the specified strategy.

    Args:
        data: dictionary of dataframes for each pair
        strategy: feature engineering strategy
        strategy_params: feature engineering strategy parameters

    Returns:
        Merged features dataframe
    '''
    if strategy == ForexFeEngStrategy.BASIC:
        return _engineer_forex_features_basic(data, **strategy_params)
    elif strategy == ForexFeEngStrategy.TA:
        return _engineer_forex_features_ta(data, **strategy_params)
    

def _engineer_forex_features_basic(
    data: Dict[str, pd.DataFrame],
    recent_returns: int
) -> pd.DataFrame:
    '''
    Basic feature engineering strategy.

    Args:
        data: collection of dataframes for each pair
        recent_returns: # of recent returns to include

    Returns:
        Features dataframe
    '''
    forex_features_df = pd.DataFrame({'<DT>' : pd.Series(dtype=FOREX_COLS['<DT>'])})

    for pair in data:
        pair_df = data[pair].drop(['<CLOSE>', '<HIGH>', '<LOW>'], axis=1)
        pair_df.rename(columns={'<OPEN>' : f'<{pair} OPEN>'}, inplace=True)

        forex_features_df = forex_features_df.merge(pair_df, on='<DT>', how='outer', suffixes=[None, None])
    
    na_dt = forex_features_df[pd.isnull(forex_features_df).any(axis=1)]['<DT>']
    if len(na_dt) > 0:
        forex_features_df = forex_features_df[forex_features_df['<DT>'] > na_dt.iloc[-1]].reset_index(drop=True)

    open_price_cols = [col for col in forex_features_df.columns if 'OPEN' in col]

    forex_features_df['<MIN SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.minute / 60).astype(np.float32)
    forex_features_df['<HOUR SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.hour / 24).astype(np.float32)
    forex_features_df['<WEEKDAY SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.weekday / 7).astype(np.float32)

    recent_returns_features = {}

    for open_price_col in open_price_cols:
    
        log_returns = np.log(forex_features_df[open_price_col] / forex_features_df[open_price_col].shift(1))
        for i in range(0, recent_returns):
            recent_returns_features[f'<{open_price_col.strip("<>")} RECENT RETURN {i+1}>'] = log_returns.shift(i)

    forex_features_df = pd.concat([forex_features_df, pd.DataFrame(recent_returns_features)], axis=1)

    forex_features_df.drop(open_price_cols, axis=1, inplace=True)
    
    forex_features_df.dropna(inplace=True)
    forex_features_df.reset_index(drop=True, inplace=True)

    return forex_features_df


def _engineer_forex_features_ta(
    data: Dict[str, pd.DataFrame],
    lags: int
) -> pd.DataFrame:
    '''
    Technical analysis feature engineering strategy.

    Args:
        data: collection of dataframes for each pair
        lags: # of indicator lags to include

    Returns:
        Features dataframe
    '''
    forex_features_df = pd.DataFrame({'<DT>' : pd.Series(dtype=FOREX_COLS['<DT>'])})

    for pair in data:
        pair_features_df = data[pair].copy()

        for price in ['HIGH', 'LOW', 'CLOSE']:
            pair_features_df[f'<{price} LAG 1>'] = pair_features_df[f'<{price}>'].shift(1)
            pair_features_df.drop(f'<{price}>', axis=1, inplace=True)

        pair_ta_features = {}

        indicator_bb = BollingerBands(
            close=pair_features_df['<CLOSE LAG 1>'], window=20, window_dev=2, fillna=True  
        )
        pair_ta_features[f'<{pair} BB HIGH INDICATOR>'] = indicator_bb.bollinger_hband_indicator()
        pair_ta_features[f'<{pair} BB LOW INDICATOR>'] = indicator_bb.bollinger_lband_indicator()
        pair_ta_features[f'<{pair} BB PERC BAND>'] = indicator_bb.bollinger_pband()
        pair_ta_features[f'<{pair} BB WIDTH BAND>'] = indicator_bb.bollinger_wband()

        indicator_atr = AverageTrueRange(
            close=pair_features_df['<CLOSE LAG 1>'], 
            high=pair_features_df['<HIGH LAG 1>'], 
            low=pair_features_df['<LOW LAG 1>'], 
            window=10, 
            fillna=True
        )
        pair_ta_features[f'<{pair} ATR>'] = indicator_atr.average_true_range()

        indicator_macd = MACD(
            close=pair_features_df['<CLOSE LAG 1>'], window_slow=26, window_fast=12, window_sign=9, fillna=True
        )
        pair_ta_features[f'<{pair} MACD DIFF>'] = indicator_macd.macd_diff()

        indicator_vortex = VortexIndicator(
            high=pair_features_df['<HIGH LAG 1>'], 
            low=pair_features_df['<LOW LAG 1>'], 
            close=pair_features_df['<CLOSE LAG 1>'],
            window=14, 
            fillna=True
        )
        pair_ta_features[f'<{pair} VORTEX DIFF>'] = indicator_vortex.vortex_indicator_diff()

        indicator_adx = ADXIndicator(
            high=pair_features_df['<HIGH LAG 1>'], 
            low=pair_features_df['<LOW LAG 1>'], 
            close=pair_features_df['<CLOSE LAG 1>'], 
            window=14, 
            fillna=True
        )
        pair_ta_features[f'<{pair} +DIR>'] = indicator_adx.adx_pos()
        pair_ta_features[f'<{pair} -DIR>'] = indicator_adx.adx_neg()

        indicator_aroon = AroonIndicator(close=pair_features_df['<CLOSE LAG 1>'], window=25, fillna=True)
        pair_ta_features[f'<{pair} AROON>'] = indicator_aroon.aroon_indicator()

        indicator_psar = PSARIndicator(
            high=pair_features_df['<HIGH LAG 1>'] ,
            low=pair_features_df['<LOW LAG 1>'] ,
            close=pair_features_df['<CLOSE LAG 1>'],
            step=0.02,
            max_step=0.20,
            fillna=True,
        )
        pair_ta_features[f'<{pair} PSAR UP INDICATOR>'] = indicator_psar.psar_up_indicator()
        pair_ta_features[f'<{pair} PSAR DOWN INDICATOR>'] = indicator_psar.psar_down_indicator()

        indicator_rsi = RSIIndicator(
            close=pair_features_df['<CLOSE LAG 1>'], window=14, fillna=True
        )
        pair_ta_features[f'<{pair} RSI>'] = indicator_rsi.rsi()

        indicator_srsi = StochRSIIndicator(
            close=pair_features_df['<CLOSE LAG 1>'], window=14, smooth1=3, smooth2=3, fillna=True
        )
        pair_ta_features[f'<{pair} STOCH RSI K>'] = indicator_srsi.stochrsi_k()

        indicator_uo = UltimateOscillator(
            high=pair_features_df['<HIGH LAG 1>'],
            low=pair_features_df['<LOW LAG 1>'],
            close=pair_features_df['<CLOSE LAG 1>'],
            window1=7,
            window2=14,
            window3=28,
            weight1=4.0,
            weight2=2.0,
            weight3=1.0,
            fillna=True
        )
        pair_ta_features[f'<{pair} UO>'] = indicator_uo.ultimate_oscillator()

        for indicator, l in itertools.product([
            'BB PERC BAND',
            'BB WIDTH BAND',
            'ATR',
            'MACD DIFF',
            'VORTEX DIFF',
            '+DIR',
            '-DIR',
            'AROON',
            'RSI',
            'STOCH RSI K',
            'UO'
        ], range(lags)):
            pair_ta_features[f'<{pair} {indicator} LAG {l+1}>'] = pair_ta_features[f'<{pair} {indicator}>'].shift(l+1)

        pair_features_df = pd.concat([pair_features_df, pd.DataFrame(pair_ta_features)], axis=1)

        pair_features_df.drop(['<OPEN>', '<HIGH LAG 1>', '<LOW LAG 1>', '<CLOSE LAG 1>'], axis=1, inplace=True)
        
        forex_features_df = forex_features_df.merge(pair_features_df, on='<DT>', how='outer', suffixes=[None, None])

    na_dt = forex_features_df[pd.isnull(forex_features_df).any(axis=1)]['<DT>']
    if len(na_dt) > 0:
        forex_features_df = forex_features_df[forex_features_df['<DT>'] > na_dt.iloc[-1]].reset_index(drop=True)
    
    forex_features_df.dropna(inplace=True)
    forex_features_df.reset_index(drop=True, inplace=True)
    
    return forex_features_df

