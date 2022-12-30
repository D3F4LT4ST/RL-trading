import gym
import pandas as pd
from ._envs import ForexEnv

class TrnOrEvalForexWrapper(gym.Wrapper):

    def __init__(self, 
        env: ForexEnv,
        target_prices_df_eval: pd.DataFrame,
        features_df_eval: pd.DataFrame,
        eval: bool=False
    ):
        super().__init__(env)

        if eval:
            setattr(env, 'target_prices_df', target_prices_df_eval)
            setattr(env, 'features_df', features_df_eval)
