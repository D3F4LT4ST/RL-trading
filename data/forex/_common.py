import os
import numpy as np
from enum import Enum

FOREX_DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), '../../Data/Forex'))

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
