from enum import IntEnum

class Actions(IntEnum):
    '''
    Basic trading actions.
    '''
    SELL = -1
    CLOSE = 0
    BUY = 1
    
class Positions(IntEnum):
    '''
    Basic trading positions.
    '''
    SHORT = -1
    NONE = 0
    LONG = 1