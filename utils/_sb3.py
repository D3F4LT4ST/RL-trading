import sys
from tqdm import tqdm
from stable_baselines3.common.callbacks import ProgressBarCallback

class TextProgressBarCallback(ProgressBarCallback):

    def __init__(self) -> None:
        super().__init__()

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.locals['total_timesteps'] - self.model.num_timesteps)
    
    def _on_training_end(self) -> None:
        sys.stderr.write('\n')
        super()._on_training_end()

