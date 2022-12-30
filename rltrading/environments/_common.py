from enum import IntEnum

class Actions(IntEnum):
    SELL = -1
    CLOSE = 0
    BUY = 1
    
class Positions(IntEnum):
    SHORT = -1
    NONE = 0
    LONG = 1