from abc import ABC, abstractmethod
import numpy as np

class BasePolicy(ABC):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    @abstractmethod
    def get_action(self, state: np.ndarray):
        pass