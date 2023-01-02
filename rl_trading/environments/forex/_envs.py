import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from typing import Set, List, Dict, Tuple
from .._common import Actions, Positions
from ._components import (
    ForexOrderStrategy,
    ForexMarketOrderStrategy,
    ForexRewardStrategy, 
    ForexTradingCostsStrategy
)

class ForexEnv(gym.Env, ABC):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, 
        target_prices_df: pd.DataFrame, 
        features_df: pd.DataFrame,
        portfolio_value: float,
        order_strategy: ForexOrderStrategy,
        reward_strategy: ForexRewardStrategy,
        trading_costs_strategy: ForexTradingCostsStrategy,
        include_in_obs: List[str]=[]
    ):
        self._target_prices_df = target_prices_df
        self._features_df = features_df
        self._init_portfolio_value = portfolio_value
        self._order_strategy = order_strategy
        self._reward_strategy = reward_strategy
        self._trading_costs_strategy = trading_costs_strategy
        self._include_in_obs = include_in_obs

        obs_space_len = features_df.shape[1] + len(include_in_obs)

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * obs_space_len, dtype=np.float32), 
            high=np.array([np.inf] * obs_space_len, dtype=np.float32)
        )

        self._order_strategy.env = self
        self._reward_strategy.env = self
        self._trading_costs_strategy.env = self

        self._reset()

    @property
    def target_prices_df(self) -> pd.DataFrame:
        return self._target_prices_df

    @target_prices_df.setter
    def target_prices_df(self, target_prices_df):
        self._target_prices_df = target_prices_df

    @property
    def features_df(self) -> pd.DataFrame:
        return self._features_df

    @features_df.setter
    def features_df(self, features_df):
        self._features_df = features_df

    @property
    def history(self) -> Dict[str, List]:
        return self._history

    def _reset(self):
        self._done = False
        self._portfolio_value = self._init_portfolio_value
        self._t = 0
        self._reward = 0
        self._history = {
            'portfolio_value' : [],
            'reward' : [],
        }
        
    def reset(self) -> np.ndarray:
        self._reset()
        self._update_history()
        return self._get_observation()

    def _update_history(self):
        self._history['portfolio_value'].append(self._portfolio_value)
        self._history['reward'].append(self._reward)
        
    def _get_observation(self) -> np.ndarray:
        return np.concatenate([
            self._features_df.iloc[self._t].values,
            [float(getattr(self, f'_{attr}')) for attr in self._include_in_obs]
        ], dtype=np.float32)

    def _step(self, action: int):
        self._t += 1
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._step(action)
        self._update_history()

        return self._get_observation(), self._reward, self._done, {}

    def render(
        self,
        mode: str='human',
        start_t: int=0,
        end_t: int=None
    ):
        if not end_t: end_t = self._t

        plt.figure(figsize=(14,5))
        plt.title('Portfolio Return vs Market Return')
        plt.plot(
            [t for t in range(start_t, end_t + 1)], 
            np.array(self._history['portfolio_value'][start_t : end_t + 1]) / self._init_portfolio_value, 
            label='Portfolio'
        )
        plt.plot(
            [t for t in range(start_t, end_t + 1)], 
            self._target_prices_df.loc[start_t : end_t, '<CLOSE>'] / self._target_prices_df.loc[0, '<CLOSE>'], 
            label='Market'
        )
        plt.legend()
        plt.show()


class ForexEnvBasic(ForexEnv):

    def __init__(self, 
        target_prices_df: pd.DataFrame, 
        features_df: pd.DataFrame, 
        portfolio_value: float,
        allowed_actions: Set[Actions],
        market_order_strategy: ForexMarketOrderStrategy,
        reward_strategy: ForexRewardStrategy,
        trading_costs_strategy: ForexTradingCostsStrategy,
        include_in_obs: List[str]=[],
    ):
        super().__init__(
            target_prices_df, 
            features_df, 
            portfolio_value,
            market_order_strategy,
            reward_strategy,
            trading_costs_strategy ,
            include_in_obs,
        )

        self.action_space = gym.spaces.Discrete(len(allowed_actions))

        self._position_color_map = {
            Positions.SHORT: 'red',
            Positions.NONE: 'grey',
            Positions.LONG: 'green'
        }

        if allowed_actions == {Actions.CLOSE, Actions.BUY}:
            self._action_map = lambda x: x
        elif allowed_actions == {Actions.CLOSE, Actions.SELL}:
            self._action_map = lambda x: -x
        elif allowed_actions == {Actions.SELL, Actions.BUY}:
            self._action_map = lambda x: np.sign(x - 0.5)
        elif allowed_actions == {Actions.SELL, Actions.CLOSE, Actions.BUY}:
            self._action_map = lambda x: x - 1

        self._reset()

    def _reset(self):
        super()._reset()
        self._action = Actions.CLOSE
        self._position = Positions.NONE
        self._trade = False
        self._last_trade_price = self._target_prices_df.loc[self._t, '<OPEN>'] 
        self._order_size = 0
        self._trading_costs = 0
        self._history['position'] = []
        self._history['trading_costs'] = []
        self._history['order_size'] = []
        self._history['last_trade_price'] = []

    def _update_history(self):
        super()._update_history()
        self._history['position'].append(self._position)
        self._history['trading_costs'].append(self._trading_costs)
        self._history['order_size'].append(self._order_size)
        self._history['last_trade_price'].append(self._last_trade_price)

    def _step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        super()._step(action)

        self._action = self._action_map(action)
        self._trade = self._position.value != self._action

        open_price = self._target_prices_df.loc[self._t, '<OPEN>']
        close_price = self._target_prices_df.loc[self._t, '<CLOSE>']
        prev_close_price = self._target_prices_df.loc[self._t - 1, '<CLOSE>']

        self._portfolio_value += self._position.value * self._order_size * (open_price - prev_close_price) / self._last_trade_price
        
        if self._trade: 
            self._last_trade_price = open_price
            self._order_size = self._order_strategy.compute_order_size()
            
        self._portfolio_value += self._action * self._order_size * (close_price - open_price) / self._last_trade_price
        
        self._position = Positions(self._action)

        self._trading_costs = self._trading_costs_strategy.compute_costs()
        self._portfolio_value -= self._trading_costs

        self._reward = self._reward_strategy.compute_reward()

        if self._t == len(self._target_prices_df) - 1:
            self._done = True

    def render(
            self, 
            mode: str='human',
            start_t: int=0,
            end_t: int=None,
            show_returns: bool=True,
            show_trades: bool=True
        ):
        if show_returns:
            super().render(mode, start_t, end_t)

        if not end_t: end_t = self._t
        if not show_trades: return

        plt.figure(figsize=(14,5))
        plt.title('Trades')
        plt.plot(
            [t for t in range(start_t, end_t + 1)], 
            self._target_prices_df.loc[start_t : end_t, '<OPEN>'], 
            label='Open'
        )
        plt.plot(
           [t for t in range(start_t, end_t + 1)], 
            self._target_prices_df.loc[start_t : end_t, '<CLOSE>'], 
            label='Close'
        )
        plt.scatter(
            [t for t in range(start_t, end_t + 1)], 
            self._target_prices_df.loc[start_t : end_t, '<OPEN>'], 
            color=list(
                map(lambda position: self._position_color_map[position], 
                self._history['position'][start_t : end_t + 1])
            ),
            s=80
        )
        plt.legend()
        plt.show()


