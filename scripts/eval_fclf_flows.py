import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    load_embeddings_splits,
    get_attribute_indices,
    select_attributes,
)
from fclf_model import FCLFConfig, ConditionalVectorField, integrate_flow


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, attrs: torch.Tensor):
        assert embeddings.shape[0] == attrs.shape[0]
        self.embeddings = embeddings
        self.attrs = attrs

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.attrs[idx]


def load_trained_model(
    ckpt_path: str,
    num_attributes: int = 5,
    embedding_dim: int = 512,
    hidden_dim: int = 256,
    projection_radius: float = 1.0,
    alpha: float = 0.1,
) -> ConditionalVectorField:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = FCLFConfig(
        embedding_dim=cfg_dict.get("embedding_dim", embedding_dim),
        num_attributes=cfg_dict.get("num_attributes", num_attributes),
        hidden_dim=cfg_dict.get("hidden_dim", hidden_dim),
        projection_radius=cfg_dict.get("projection_radius", projection_radius),
        alpha=cfg_dict.get("alpha", alpha),
    )
    model = ConditionalVectorField(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def compute_trajectory_shifts(
    traj: torch.Tensor,
) -> torch.Tensor:
    """
    Given a trajectory tensor of shape (T+1, B, D), compute per-step
    squared shifts ||z_{k+1} - z_k||^2 averaged over batch.
    Returns tensor of shape (T,) with per-step means.
    """
    diffs = traj[1:] - traj[:-1]  # (T, B, D)
    sq = diffs.pow(2).sum(dim=-1).mean(dim=1)  # (T,)
    return sq


def evaluate_multi_step_stability(
    ckpt_path: str,
    embedding_dir: str,
    num_steps: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    target_attrs = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
    attr_indices = get_attribute_indices(train_split.attr_names, target_attrs)

    val_attrs = select_attributes(val_split, attr_indices)
    val_ds = EmbeddingDataset(val_split.embeddings, val_attrs)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = load_trained_model(ckpt_path)
    model.to(device)
    model.eval()

    all_step_shifts: List[torch.Tensor] = []

    with torch.no_grad():
        for z_batch, y_batch in val_loader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            _, traj = integrate_flow(
                model,
                z0=z_batch,
                y=y_batch,
                num_steps=num_steps,
                return_trajectory=True,
            )
            assert traj is not None
            step_shifts = compute_trajectory_shifts(traj)  # (T,)
            all_step_shifts.append(step_shifts.cpu())

    all_step_shifts_tensor = torch.stack(all_step_shifts, dim=0)  # (num_batches, T)
    mean_shifts = all_step_shifts_tensor.mean(dim=0)  # (T,)

    print(f"=== Multi-step stability diagnostics ({os.path.basename(ckpt_path)}) ===")
    for k in range(num_steps):
        print(f"step {k+1:02d}: mean ||Δz||^2 = {mean_shifts[k].item():.6f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stability_summary.txt"
    lines = [
        f"Checkpoint: {ckpt_path}",
        f"num_steps: {num_steps}",
        "",
        "=== Multi-step stability diagnostics ===",
    ]
    for k in range(num_steps):
        lines.append(f"step {k+1:02d}: mean ||Δz||^2 = {mean_shifts[k].item():.6f}")
    summary_path.write_text("\n".join(lines))
    print(f"Saved stability summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-step stability of FCLF flows.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join("checkpoints", "default", "fclf_last.pt"),
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory containing precomputed embeddings.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of flow steps to unroll.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/stability",
        help="Directory to store stability summaries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_multi_step_stability(
        ckpt_path=args.checkpoint_path,
        embedding_dir=args.embedding_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
    )


