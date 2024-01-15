import numpy as np
from enum import Enum

FOREX_PAIRS = [
    'AUDJPY',
    'AUDUSD',
    'CHFJPY',
    'EURCAD',
    'EURCHF',
    'EURGBP',
    'EURJPY',
    'EURUSD',
    'GBPCHF',
    'GBPJPY',
    'GBPUSD',
    'NZDJPY',
    'NZDUSD',
    'USDCAD',
    'USDCHF',
    'USDJPY'    
]

FOREX_COLS = {
    '<DT>' : 'datetime64[ns]', 
    '<OPEN>' : np.float32, 
    '<HIGH>' : np.float32, 
    '<LOW>' : np.float32, 
    '<CLOSE>' : np.float32
}

class ForexDataSource(Enum):
    '''
    Forex data sources.
    '''
    HISTDATA = 'HistData'
    FOREXTESTER = 'ForexTester'
