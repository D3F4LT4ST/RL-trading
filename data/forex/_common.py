import os
import numpy as np
from enum import Enum
from config import DATA_PATH

FOREX_DATA_PATH = os.path.join(DATA_PATH, 'Forex')

FOREX_COLS = {
    '<DT>' : 'datetime64[ns]', 
    '<OPEN>' : np.float32, 
    '<HIGH>' : np.float32, 
    '<LOW>' : np.float32, 
    '<CLOSE>' : np.float32
}

class ForexDataSource(Enum):
    HISTDATA = 'HistData'
    FOREXTESTER = 'ForexTester'
