## distribution
from __future__ import annotations
from . import Sampleable, Density
from typing import Self

import torch
import torch.distributions as D

import numpy as np

class Gaussian(Sampleable, Density, torch.nn.Module):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
    
    @property
    def dim(self) -> int:
        return self.mean.shape[0]
    
    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov)
    
    def sample(self, n: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((n,)))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)
    
    @classmethod
    def isotropic(cls, dim: int, std: float = 1.0) -> Self:
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * (std ** 2)
        return cls(mean, cov)
    
class GaussianMixture(Sampleable, Density, torch.nn.Module):
    """GMM of 2-dim Gaussians"""
    def __init__(
        self, 
        means: torch.Tensor, 
        covs: torch.Tensor,
        weights: torch.Tensor,
        ):
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('covs', covs)
        self.register_buffer('weights', weights)
    
    @property
    def dim(self) -> int:
        return self.means.shape[1]
    
    @property
    def distribution(self) -> D.MixtureSameFamily:
        base_dist = D.MultivariateNormal(self.means, self.covs)
        mixture_dist = D.Categorical(self.weights)
        return D.MixtureSameFamily(mixture_dist, base_dist)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)
    
    def sample(self, n: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((n,)))

    @classmethod
    def rand_2D(
        cls, nmodes: int, std: float = 1.0, scale: float = 1.0, x_offset: float = 0, seed: int = 0
    ) -> Self:
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs =torch.diag_embed(torch.ones(nmodes, 2) * (std ** 2))
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
    
    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> Self:
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
    
