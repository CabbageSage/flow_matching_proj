from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Self, Tuple, Callable, List

import torch
import torch.distributions as D
from torch.func import vmap, jacrev


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib.axes._axes import Axes
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.datasets import make_moons, make_circles

from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampleable(ABC):
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:...

    @property
    @abstractmethod
    def dim(self) -> int:...
        
class Density(ABC):
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:...
    
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:...
    
