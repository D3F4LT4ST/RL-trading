import pandas as pd
import numpy as np
from ._common import ForexDataSource
from typing import Dict

def preprocess_forex_data(
    data: Dict[str, pd.DataFrame], 
    data_source: ForexDataSource,
    agg_interval: int
):
    '''
    Preprocesses forex data.

    Args:
        data: collection of dataframes for each pair
        data_source: data source
        agg_interval: aggregation interval
    
    Returns:
        Preprocesed dataframes for each pair
    '''
    if data_source == ForexDataSource.FOREXTESTER:
        return _preprocess_forextester_forex_data(data, agg_interval)


def _preprocess_forextester_forex_data(
    data: Dict[str, pd.DataFrame],
    agg_interval: int
):
    '''
    Preprocesses forex data from ForexTester.

    Args:
        data: collection of dataframes for each pair
        agg_interval: aggregation interval
    
    Returns:
        Preprocesed dataframes for each pair
    '''
    for pair in data:

        pair_full_daterange_df = pd.DataFrame(
            pd.date_range(data[pair]['<DT>'].min(), data[pair]['<DT>'].max(), freq='min'), 
            columns=['<DT>']
        )
        data[pair] = data[pair].merge(pair_full_daterange_df, how='right', on='<DT>')
        
        pair_first_nan_idx = np.where(
            (~data[pair]['<CLOSE>'].shift(1).isna()) & (data[pair]['<CLOSE>'].isna())
        )[0]
        data[pair].loc[pair_first_nan_idx, ['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']] = pd.Series(
            data[pair].iloc[pair_first_nan_idx - 1]['<CLOSE>'].values,
            index=pair_first_nan_idx
        )
        data[pair].fillna(method='ffill', inplace=True)
        
        data[pair]['<DT>'] = data[pair]['<DT>'].apply(
            lambda dt: dt.replace(minute=(dt.minute // agg_interval) * agg_interval)
        )
        data[pair] = data[pair].groupby('<DT>').agg({
            '<OPEN>' : lambda group: group.iloc[0],
            '<HIGH>' : np.max, 
            '<LOW>' : np.min, 
            '<CLOSE>' : lambda group: group.iloc[-1]
        }).reset_index()

