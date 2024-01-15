from tqdm import tqdm
from stable_baselines3.common.callbacks import ProgressBarCallback

class TextProgressBarCallback(ProgressBarCallback):
    '''
    Callback for displaying tqdm progress bar in text.
    '''
    def __init__(self) -> None:
        super().__init__()

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.locals['total_timesteps'] - self.model.num_timesteps)
    

