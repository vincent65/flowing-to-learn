import argparse
import os
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import yaml

from celeba_embeddings import (
    load_embeddings_splits,
    get_attribute_indices,
    select_attributes,
)
from fclf_model import FCLFConfig, ConditionalVectorField, tangent_sphere_step


class EmbeddingDataset(Dataset):
    """
    Simple dataset over precomputed CLIP embeddings and selected attributes.
    """

    def __init__(self, embeddings: torch.Tensor, attrs: torch.Tensor):
        assert embeddings.shape[0] == attrs.shape[0]
        self.embeddings = embeddings
        self.attrs = attrs

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.attrs[idx]


def supervised_contrastive_loss_multi_label(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Multi-label supervised contrastive loss on flowed embeddings.

    z: (B, D) – flowed embeddings
    y: (B, K) – binary attribute vectors {0,1}
    """
    z = nn.functional.normalize(z, dim=-1)
    B = z.size(0)

    # Similarity matrix
    sim = torch.matmul(z, z.t()) / temperature  # (B, B)

    # Mask out self-similarities
    logits_mask = torch.ones_like(sim) - torch.eye(B, device=sim.device)
    sim = sim * logits_mask

    # Positive mask: samples with identical attribute vectors
    y_i = y.unsqueeze(1)  # (B, 1, K)
    y_j = y.unsqueeze(0)  # (1, B, K)
    pos_mask = (y_i == y_j).all(dim=-1).float() * logits_mask  # (B, B)

    # Compute log-probabilities
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)  # (B,)
    # Avoid division by zero; samples without positives do not contribute
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    valid = pos_count > 0
    if valid.sum() == 0:
        # Degenerate minibatch (all labels unique) – return zero to avoid NaNs
        return torch.zeros((), device=z.device)

    loss = -(mean_log_prob_pos[valid]).mean()
    return loss


def train_one_epoch(
    model: ConditionalVectorField,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    lambda_contrastive: float,
    lambda_identity: float,
    lambda_field_norm: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_con = 0.0
    total_reg = 0.0
    n_samples = 0

    for z_batch, y_batch in dataloader:
        z_batch = z_batch.to(device)
        y_batch = y_batch.to(device)

        z_next, v_tan = tangent_sphere_step(model, z_batch, y_batch, alpha=alpha)

        L_con = supervised_contrastive_loss_multi_label(z_next, y_batch, temperature=temperature)

        # Identity / manifold preservation term
        L_id = (z_next - z_batch).pow(2).sum(dim=-1).mean()

        # Field-norm regularization to discourage huge velocities
        L_field = v_tan.pow(2).sum(dim=-1).mean()

        loss = (
            lambda_contrastive * L_con
            + lambda_identity * L_id
            + lambda_field_norm * L_field
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = z_batch.size(0)
        total_loss += loss.item() * bs
        total_con += L_con.item() * bs
        total_reg += (L_id.item() + L_field.item()) * bs
        n_samples += bs

    mean_loss = total_loss / n_samples
    mean_con = total_con / n_samples
    mean_reg = total_reg / n_samples
    return mean_loss, mean_con, mean_reg


def evaluate_identity_shift(
    model: ConditionalVectorField,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
) -> float:
    """
    Simple stability diagnostic: average squared shift ||z_next - z||^2.
    """
    model.eval()
    total_shift = 0.0
    n_samples = 0

    with torch.no_grad():
        for z_batch, y_batch in dataloader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            z_next, _ = tangent_sphere_step(model, z_batch, y_batch, alpha=alpha)
            shift = (z_next - z_batch).pow(2).sum(dim=-1).mean()
            bs = z_batch.size(0)
            total_shift += shift.item() * bs
            n_samples += bs

    return total_shift / n_samples


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    path: str,
    model: ConditionalVectorField,
    optimizer: torch.optim.Optimizer,
    cfg: FCLFConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
        },
        path,
    )


def maybe_resume_training(
    checkpoint_path: str,
    model: ConditionalVectorField,
    optimizer: torch.optim.Optimizer,
) -> int:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found; cannot resume."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"[INFO] Resuming training from epoch {start_epoch} using {checkpoint_path}")
    return start_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FCLF on CelebA CLIP embeddings.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fclf_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with train/val/test embedding .pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/default",
        help="Directory to store checkpoints for this run.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Full path to the latest-checkpoint file. "
             "If omitted, defaults to <output_dir>/fclf_last.pt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --checkpoint_path if it exists.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save an additional checkpoint every N epochs.",
    )
    args = parser.parse_args()

    cfg_yaml = load_yaml_config(args.config)

    model_cfg = cfg_yaml.get("model", {})
    training_cfg = cfg_yaml.get("training", {})
    loss_cfg = cfg_yaml.get("loss", {})

    embedding_dim = model_cfg.get("embedding_dim", 512)
    num_attributes = model_cfg.get("num_attributes", 5)
    hidden_dim = model_cfg.get("hidden_dim", 256)
    projection_radius = model_cfg.get("projection_radius", 1.0)

    num_epochs = training_cfg.get("num_epochs", 50)
    batch_size = training_cfg.get("batch_size", 512)
    learning_rate = training_cfg.get("learning_rate", 1e-4)
    alpha = training_cfg.get("alpha", 0.1)

    temperature = loss_cfg.get("temperature", 0.1)
    lambda_contrastive = loss_cfg.get("lambda_contrastive", 0.7)
    lambda_identity = loss_cfg.get("lambda_identity", 0.3)

    # We use lambda_curl as a proxy for field-norm regularization strength.
    lambda_field_norm = loss_cfg.get("lambda_curl", 0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading precomputed embeddings...")
    train_split, val_split, _ = load_embeddings_splits(args.embedding_dir)

    target_attrs = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
    attr_indices = get_attribute_indices(train_split.attr_names, target_attrs)

    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)

    # Datasets / loaders
    train_ds = EmbeddingDataset(train_split.embeddings, train_attrs)
    val_ds = EmbeddingDataset(val_split.embeddings, val_attrs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("[INFO] Building vector field model...")
    fclf_cfg = FCLFConfig(
        embedding_dim=embedding_dim,
        num_attributes=num_attributes,
        hidden_dim=hidden_dim,
        projection_radius=projection_radius,
        alpha=alpha,
    )
    model = ConditionalVectorField(fclf_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else os.path.join(args.output_dir, "fclf_last.pt")
    )

    start_epoch = 0
    if args.resume:
        start_epoch = maybe_resume_training(checkpoint_path, model, optimizer)
        if start_epoch >= num_epochs:
            print(
                f"[INFO] Checkpoint already at epoch {start_epoch} >= num_epochs; nothing to do."
            )
            return

    print(
        f"[INFO] Training for {num_epochs - start_epoch} additional epochs "
        f"(total target {num_epochs}) on device {device} with batch_size={batch_size}"
    )
    for epoch in range(start_epoch + 1, num_epochs + 1):
        mean_loss, mean_con, mean_reg = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=temperature,
            lambda_contrastive=lambda_contrastive,
            lambda_identity=lambda_identity,
            lambda_field_norm=lambda_field_norm,
        )

        avg_shift = evaluate_identity_shift(
            model=model,
            dataloader=val_loader,
            device=device,
            alpha=alpha,
        )

        print(
            f"[Epoch {epoch:03d}/{num_epochs}] "
            f"loss={mean_loss:.4f} | con={mean_con:.4f} | reg={mean_reg:.4f} | "
            f"avg_val_shift={avg_shift:.4f}"
        )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            cfg=fclf_cfg,
            epoch=epoch,
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_ckpt = os.path.join(
                args.output_dir,
                f"fclf_epoch_{epoch:03d}.pt",
            )
            save_checkpoint(
                path=epoch_ckpt,
                model=model,
                optimizer=optimizer,
                cfg=fclf_cfg,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()


