from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FCLFConfig:
    embedding_dim: int = 512
    num_attributes: int = 5
    hidden_dim: int = 256
    projection_radius: float = 1.0
    alpha: float = 0.1  # flow step size


class ConditionalVectorField(nn.Module):
    """
    v_ω(z, y): R^d × R^k → R^d

    - z: CLIP embedding (already L2-normalized, but we enforce on a sphere)
    - y: attribute vector (e.g., binary of length num_attributes)
    """

    def __init__(self, cfg: FCLFConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.embedding_dim + cfg.hidden_dim

        # Embed attributes into a hidden space
        self.attr_mlp = nn.Sequential(
            nn.Linear(cfg.num_attributes, cfg.hidden_dim),
            nn.GELU(),
        )

        # MLP over [z, y_embed]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

        # Global scale to keep velocities bounded
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) CLIP embeddings
            y: (B, K) attribute vectors in {0,1}
        Returns:
            v_raw: (B, D) unconstrained vector field values
        """
        if y.dtype != torch.float32:
            y = y.float()
        y_embed = self.attr_mlp(y)
        inp = torch.cat([z, y_embed], dim=-1)
        delta = self.mlp(inp)
        v = torch.tanh(delta) * self.scale  # bound magnitude
        return v


def tangent_sphere_step(
    model: ConditionalVectorField,
    z: torch.Tensor,
    y: torch.Tensor,
    alpha: Optional[float] = None,
    projection_radius: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single Euler step on the unit sphere with a tangential vector field.

    Args:
        model: ConditionalVectorField
        z: (B, D) current embeddings
        y: (B, K) attributes
        alpha: step size (defaults to cfg.alpha)
        projection_radius: sphere radius (defaults to cfg.projection_radius)

    Returns:
        z_next: (B, D) next embeddings (L2-normalized to given radius)
        v_tan:  (B, D) tangential vector field used for the update
    """
    cfg = model.cfg
    if alpha is None:
        alpha = cfg.alpha
    if projection_radius is None:
        projection_radius = cfg.projection_radius

    # Ensure points lie on the sphere
    z_hat = F.normalize(z, dim=-1) * projection_radius

    v_raw = model(z_hat, y)
    # Project to tangent space of the sphere: v_tan = v - (v·z_hat) z_hat
    dot = (v_raw * z_hat).sum(dim=-1, keepdim=True)
    v_tan = v_raw - dot * z_hat

    z_next = z_hat + alpha * v_tan
    z_next = F.normalize(z_next, dim=-1) * projection_radius
    return z_next, v_tan


def integrate_flow(
    model: ConditionalVectorField,
    z0: torch.Tensor,
    y: torch.Tensor,
    num_steps: int,
    alpha: Optional[float] = None,
    projection_radius: Optional[float] = None,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Multi-step Euler integration of the flow on the sphere.

    Args:
        model: ConditionalVectorField
        z0: (B, D) initial embeddings
        y: (B, K) attributes (fixed over the trajectory)
        num_steps: number of flow steps to apply
        alpha: step size (defaults to cfg.alpha)
        projection_radius: sphere radius
        return_trajectory: if True, also return a tensor of shape (num_steps+1, B, D)
                           containing all intermediate embeddings (including z0).

    Returns:
        z_T: final embeddings after num_steps
        traj: optional trajectory tensor or None
    """
    zs = []
    z = z0
    if return_trajectory:
        zs.append(z0.detach().clone())

    for _ in range(num_steps):
        z, _ = tangent_sphere_step(
            model,
            z,
            y,
            alpha=alpha,
            projection_radius=projection_radius,
        )
        if return_trajectory:
            zs.append(z.detach().clone())

    if return_trajectory:
        traj = torch.stack(zs, dim=0)  # (T+1, B, D)
    else:
        traj = None
    return z, traj




