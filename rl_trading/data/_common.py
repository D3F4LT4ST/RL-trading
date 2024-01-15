import pandas as pd
from datetime import datetime
from typing import Tuple

def train_val_eval_split(
    df: pd.DataFrame, 
    trn_start_date: datetime, 
    trn_end_date: datetime, 
    val_start_date: datetime, 
    val_end_date: datetime, 
    eval_start_date: datetime, 
    eval_end_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Splits the dataframe into consecutive train, validation and evaluation sets.

    Args:
        trn_start_date: train set start date
        trn_end_date:  train set end date
        val_start_date: validation set start date
        val_end_date: validation set end date
        eval_start_date: evaluation set start date
        eval_end_date: evaluation set end date
    '''
    df_trn = df[(df['<DT>'] >= trn_start_date) & (df['<DT>'] < trn_end_date)].reset_index(drop=True)
    df_val = df[(df['<DT>'] >= val_start_date) & (df['<DT>'] < val_end_date)].reset_index(drop=True)
    df_eval = df[(df['<DT>'] >= eval_start_date) & (df['<DT>'] < eval_end_date)].reset_index(drop=True)

    return df_trn, df_val, df_eval