import os
import pandas as pd
from ._common import (
    FOREX_COLS,
    ForexDataSource
)
from typing import List, Dict

def load_processed_forex_data(
    data_path: str,
    data_source: ForexDataSource, 
    pairs: List[str],
    version: str=None
) -> Dict[str, pd.DataFrame]:

    proc_data_path = f'{data_path}/Forex/{data_source.value}/Processed'
    data = {}

    for pair in pairs:
        pair_data_path = (
            f'{proc_data_path}/' +
            (f'{version}/' if version else '') +
            f'{pair}.pkl'
        )
        pair_df = pd.read_pickle(pair_data_path)
        data[pair] = pair_df

    return data


def load_raw_forex_data(
    data_path: str,
    data_source: ForexDataSource, 
    pairs: List[str]
) -> Dict[str, pd.DataFrame]:
    if data_source == ForexDataSource.FOREXTESTER:
        return _load_raw_forextester_forex_data(data_path, pairs)
    elif data_source == ForexDataSource.HISTDATA:
        return _load_raw_histdata_forex_data(data_path, pairs)


def _load_raw_histdata_forex_data(
    data_path: str,
    pairs: List[str]
) -> Dict[str, pd.DataFrame]:

    histdata_raw_data_path = f'{data_path}/Forex/HistData/Raw'
    histdata_data = {}

    for pair in pairs:

        pair_data_path = f'{histdata_raw_data_path}/{pair}'
        pair_df = pd.DataFrame(columns=FOREX_COLS.keys())

        for fragment_dir in os.listdir(pair_data_path):
            if 'HISTDATA' not in fragment_dir: continue

            fragment_dir_path = f'{pair_data_path}/{fragment_dir}'
            fragment_file = [
                file for file in os.listdir(fragment_dir_path) if file.endswith('.csv')
            ][0]      
            fragment_df = pd.read_csv(
                f'{fragment_dir_path}/{fragment_file}', delimiter=';', header=None
            )
            fragment_df.drop([5], axis=1, inplace=True)
            fragment_df.columns = FOREX_COLS.keys()

            pair_df = pd.concat([pair_df, fragment_df])

        pair_df = pair_df.astype(FOREX_COLS)
        pair_df = pair_df.sort_values('<DT>').reset_index(drop=True)   
        histdata_data[pair] = pair_df

    return histdata_data

def _load_raw_forextester_forex_data(
    data_path: str,
    pairs: List[str]
) -> Dict[str, pd.DataFrame]:

    forextester_raw_data_path = f'{data_path}/Forex/ForexTester/Raw'
    forextester_data = {}

    for pair in pairs:
        pair_df = pd.read_csv(
            f'{forextester_raw_data_path}/{pair}.txt', 
            dtype={'<DTYYYYMMDD>' : str, '<TIME>' : str}
        )
        pair_df['<DT>'] = pd.to_datetime(pair_df['<DTYYYYMMDD>'] + ' ' + pair_df['<TIME>']) 
        pair_df = pair_df.astype(FOREX_COLS)
        pair_df.drop(['<DTYYYYMMDD>', '<TIME>', '<TICKER>', '<VOL>'], axis=1, inplace=True)
        pair_df = pair_df[list(pair_df.columns)[-1:] + list(pair_df.columns)[:-1]]
        pair_df = pair_df.sort_values('<DT>').reset_index(drop=True)   
        forextester_data[pair] = pair_df

    return forextester_data
