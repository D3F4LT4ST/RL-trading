import gym
import numpy as np
import pandas as pd
from gym.utils.seeding import np_random
from typing import List, Tuple, Dict, Any, Type
from ._envs import ForexEnv

class TrnOrEvalForexWrapper(gym.Wrapper):
    '''
    Gym wrapper, switches between training and evaluation modes.
    '''
    def __init__(
        self, 
        env: ForexEnv,
        target_prices_df_eval: pd.DataFrame,
        features_df_eval: pd.DataFrame,
        eval: bool=False,
        trn_wrappers: Dict[Type[gym.Wrapper], Dict[str, Any]]={},
        eval_wrappers: Dict[Type[gym.Wrapper], Dict[str, Any]]={}
    ):
        '''
        Args:
            env: forex environment
            target_prices_df_eval: evaluation target pair dataframe
            features_df_eval: evaluation features dataframe,
            eval: evaluation mode?
            trn_wrappers: train environment wrappers
            eval_wrappers: evaluation environment wrappers
        '''
        super().__init__(env)

        if eval:
            self.env.target_prices_df = target_prices_df_eval
            self.env.features_df = features_df_eval
            wrappers = eval_wrappers
        else:
            wrappers = trn_wrappers

        for wrapper_class in wrappers.keys():
            self.env = wrapper_class(self.env, **wrappers[wrapper_class])


class RandomEpisodeForexWrapper(gym.Wrapper):
    '''
    Gym wrapper, trims the episode to desired length starting at random timestamp. 
    '''
    def __init__(
        self, 
        env: ForexEnv,
        episode_length: int
    ) -> None:
        '''
        Args:
            env: forex environment
            episode_length: episode length
        '''
        super().__init__(env)

        self._episode_length = episode_length
        self._episode_t = 0
        self._env_start_t = 0
        self._rng = np.random.RandomState()

    def reset(self) -> np.ndarray:
        self._episode_t = 0
        self._env_start_t = self._rng.randint(0, len(self.env) - self._episode_length)

        return self.env.reset(self._env_start_t )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._episode_t += 1
        obs, reward, done, info = self.env.step(action)

        if self._episode_t == self._episode_length:
            done = True
        
        return obs, reward, done, info 

    def seed(self, seed: int) -> List[int]:
        self._rng, seed = np_random(seed)
        return [seed]

    def render(
        self,
        mode: str='human',
        start_t: int=0,
        end_t: int=None,
        **kwargs
    ):
        if not end_t: end_t = self._episode_t
        self.env.render(
            mode, 
            self._env_start_t + start_t,
            self._env_start_t + end_t,
            **kwargs
        )
