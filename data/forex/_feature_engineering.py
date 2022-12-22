import pandas as pd
import numpy as np
from typing import Dict
from enum import Enum
from ._common import FOREX_COLS

class ForexFeEngStrategy(Enum):
    STRATEGY1 = 0 # "Financial Trading as a Game: A Deep Reinforcement Learning Approach" 


def engineer_forex_features(
    data: Dict[str, pd.DataFrame],
    strategy: ForexFeEngStrategy,
    strategy_params: Dict
) -> pd.DataFrame:
    if strategy == ForexFeEngStrategy.STRATEGY1:
        return _engineer_forex_features_strategy1(data, **strategy_params)
    

def _engineer_forex_features_strategy1(
    data: Dict[str, pd.DataFrame],
    recent_returns: int
) -> pd.DataFrame:
    forex_features_df = pd.DataFrame({'<DT>' : pd.Series(dtype=FOREX_COLS['<DT>'])})

    for pair in data:
        pair_df = data[pair].drop(['<OPEN>', '<HIGH>', '<LOW>'], axis=1)
        pair_df.rename(columns={'<CLOSE>' : f'<{pair} CLOSE>'}, inplace=True)

        forex_features_df = forex_features_df.merge(pair_df, on='<DT>', how='outer', suffixes=[None, None])
    
    na_dt = forex_features_df[pd.isnull(forex_features_df).any(axis=1)]['<DT>']
    if len(na_dt) > 0:
        forex_features_df = forex_features_df[forex_features_df['<DT>'] > na_dt.iloc[-1]].reset_index(drop=True)

    close_price_cols = [col for col in forex_features_df.columns if 'CLOSE' in col]

    forex_features_df['<MIN SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.minute / 60).astype(np.float32)
    forex_features_df['<HOUR SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.hour / 24).astype(np.float32)
    forex_features_df['<WEEKDAY SIN>'] = np.sin(2 * np.pi * forex_features_df['<DT>'].dt.weekday / 7).astype(np.float32)

    for close_price_col in close_price_cols:
    
        log_returns = np.log(forex_features_df[close_price_col] / forex_features_df[close_price_col].shift(1))
        for i in range(0, recent_returns):
            forex_features_df[f'<{close_price_col.strip("<>")} RECENT RETURN {i+1}>'] = log_returns.shift(i)

        forex_features_df.drop(close_price_col, axis=1, inplace=True)
    
    forex_features_df.dropna(inplace=True)
    forex_features_df.reset_index(drop=True, inplace=True)

    return forex_features_df




