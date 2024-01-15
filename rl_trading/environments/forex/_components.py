import numpy as np
from abc import ABC, abstractmethod
from .._common import Actions, Positions
from typing import TYPE_CHECKING
if TYPE_CHECKING: from forex import ForexEnv

class ForexEnvComponent(ABC):
    '''
    Base class for forex environment components.
    '''
    def __init__(self) -> None:
        self._env = None
        
    @property
    def env(self) -> 'ForexEnv':
        return self._env

    @env.setter
    def env(self, env: 'ForexEnv'):
        self._env = env


class ForexOrderStrategy(ForexEnvComponent, ABC):
    '''
    Base class for forex order placing strategies.
    '''
    def __init__(self) -> None:
        super().__init__()


class ForexMarketOrderStrategy(ForexOrderStrategy):
    '''
    Base class for forex market order placing strategies.
    '''
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_order_size(self) -> float:
        '''
        Calulates order size.
        '''
        pass


class ForexMarketOrderStrategyFixed(ForexMarketOrderStrategy):
    '''
    Fixed size forex market order placing strategy.
    '''
    def __init__(self, order_size: float) -> None:
        '''
        Args:
            order_size: order size
        '''
        super().__init__()
        self._order_size = order_size

    def compute_order_size(self) -> float:
        return self._order_size
    

class ForexMarketOrderStrategyAllIn(ForexMarketOrderStrategy):
    '''
    All-in forex market order placing strategy.
    '''
    def __init__(self) -> None:
        super().__init__()

    def compute_order_size(self) -> float:
        return self._env._portfolio_value


class ForexRewardStrategy(ForexEnvComponent, ABC):
    '''
    Base class for forex reward calculation strategies.
    '''
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_reward(self) -> float:
        '''
        Calculates reward for current timestep.
        '''
        pass


class ForexRewardStrategyLogPortfolioReturn(ForexRewardStrategy):
    '''
    Log portfolio return forex reward calculation strategy.
    '''
    def __init__(self) -> None:
        super().__init__()

    def compute_reward(self) -> float:
        if self._env._t > 0:
            prev_portfolio_value = self._env.history['portfolio_value'][-1]
        else:
            prev_portfolio_value = self._env._init_portfolio_value

        if self._env._portfolio_value != prev_portfolio_value:
            return np.log(self._env._portfolio_value / prev_portfolio_value)
        else:
            return 0


class ForexRewardStrategyWeightedLogPortfolioReturns(ForexRewardStrategyLogPortfolioReturn):
    '''
    Weighted log portfolio returns forex reward calculation strategy.
    '''
    def __init__(self) -> None:
        super().__init__()
        self._position_t = 1
        self._position_t_sum = 1
        self._weighted_reward = 0

    def compute_reward(self) -> float:

        current_log_return = super().compute_reward()

        if self._env._trade:
            self._position_t = 1
            self._position_t_sum = 1
            self._weighted_reward = current_log_return
        else:
            self._position_t += 1
            self._weighted_reward = (
                (self._weighted_reward * self._position_t_sum + current_log_return * self._position_t) / 
                (self._position_t_sum + self._position_t)
            )
            self._position_t_sum += self._position_t

        return self._weighted_reward


class ForexTradingCostsStrategy(ForexEnvComponent):
    '''
    Composite base class for forex trading costs calculation strategies.
    '''
    def __init__(
        self, 
        trading_costs_strategy_inner: 'ForexTradingCostsStrategy'=None
    ) -> None:
        '''
        Args:
            trading_costs_strategy_inner: inner trading cost calculation strategy
        '''
        super().__init__()
        self._trading_costs_strategy_inner = trading_costs_strategy_inner

    def compute_costs(self) -> float:
        if self._trading_costs_strategy_inner is not None:
            return self._trading_costs_strategy_inner.compute_costs()
        else:
            return 0 


class ForexTradingCostsStrategyRelativeFee(ForexTradingCostsStrategy):
    '''
    Relative fee-based forex trading costs calculation strategy.
    '''
    def __init__(
        self, 
        fee_rate: float,
        trading_costs_strategy_inner: 'ForexTradingCostsStrategy'=None
    ) -> None:
        '''
        Args:
            fee_rate: fee rate
            trading_costs_strategy_inner: inner trading cost calculation strategy
        '''
        super().__init__(trading_costs_strategy_inner)
        self._fee_rate = fee_rate

    def compute_costs(self) -> float:
        relative_fee = 0

        if self._env._t > 0:
            prev_position = self._env.history['position'][-1]
        else:
            prev_position = Positions.NONE

        if self._env._trade:
            relative_fee = (
                self._env._order_size * 
                abs(self._env._action - prev_position) * 
                self._fee_rate
            )
        
        return super().compute_costs() + relative_fee


class ForexTradingCostsStrategySpread(ForexTradingCostsStrategy):
    '''
    Spread-based trading costs calculation strategy.
    '''
    def __init__(
        self, 
        spread: float,
        trading_costs_strategy_inner: 'ForexTradingCostsStrategy'=None,
    ) -> None:
        '''
        Args:
            spread: spread
            trading_costs_strategy_inner: inner trading cost calculation strategy
        '''
        super().__init__(trading_costs_strategy_inner)
        self._spread = spread
    
    def compute_costs(self):
        spread_costs = 0

        if self._env._trade:

            if self._env._t > 0:
                prev_position = self._env.history['position'][-1]
            else:
                prev_position = Positions.NONE

            if (
                (prev_position == Positions.NONE and self._env._action == Actions.BUY)):
                spread_costs = self._env._order_size * self._spread / self._env._last_trade_price
            elif prev_position == Positions.SHORT and self._env._action == Actions.CLOSE:
                spread_costs = self._env.history['order_size'][-1] * self._spread / self._env.history['last_trade_price'][-1]
            elif prev_position == Positions.SHORT and self._env._action == Actions.BUY:
                spread_costs = (
                    self._env.history['order_size'][-1] * self._spread / self._env.history['last_trade_price'][-1] +
                    self._env._order_size * self._spread / self._env._last_trade_price
                )

        return super().compute_costs() + spread_costs
