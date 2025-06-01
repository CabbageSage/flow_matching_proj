from __future__ import annotations

from abc import ABC, abstractmethod
from . import Sampleable, Density
import torch

from tqdm import tqdm

## gt
class ODE(ABC):
    @abstractmethod
    def drift(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:...
    
class SDE(ABC):
    @abstractmethod
    def drift(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:...
    
    @abstractmethod
    def diffusion(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:...
    

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        take one simulation step
        Args:
            xt: current state at time t, shape (batch_size, dim)
            t: current time, shape (batch_size,)
            dt: time step size, shape (batch_size,)
        Returns:
            next: next state at time t + dt, shape (batch_size, dim)
        """
    
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            dt = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, dt)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (bs, num_timesteps, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num
            _timesteps, dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        drift = self.ode.drift(xt, t)
        return xt + drift * dt
    
def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )
    
