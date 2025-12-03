import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import shuffle as sk_shuffle

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    load_embeddings_splits,
    get_attribute_indices,
    select_attributes,
)
from fclf_model import FCLFConfig, ConditionalVectorField, integrate_flow, tangent_sphere_step


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, attrs: torch.Tensor):
        assert embeddings.shape[0] == attrs.shape[0]
        self.embeddings = embeddings
        self.attrs = attrs

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.attrs[idx]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def load_trained_model(
    ckpt_path: str,
    cfg_fallback: FCLFConfig,
) -> ConditionalVectorField:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = FCLFConfig(
        embedding_dim=cfg_dict.get("embedding_dim", cfg_fallback.embedding_dim),
        num_attributes=cfg_dict.get("num_attributes", cfg_fallback.num_attributes),
        hidden_dim=cfg_dict.get("hidden_dim", cfg_fallback.hidden_dim),
        projection_radius=cfg_dict.get(
            "projection_radius", cfg_fallback.projection_radius
        ),
        alpha=cfg_dict.get("alpha", cfg_fallback.alpha),
    )
    model = ConditionalVectorField(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def make_pca_scatter_with_trajectories(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    attr_idx: int,
    attr_name: str,
    num_points: int = 3000,
    num_traj: int = 50,
    num_steps: int = 10,
    device: torch.device = torch.device("cpu"),
    output_dir: str = "notebooks/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Subsample for PCA and scatter
    X_np = _to_numpy(embeddings)
    Y_np = _to_numpy(attrs)
    X_sub, Y_sub = sk_shuffle(X_np, Y_np, random_state=0)
    X_sub = X_sub[:num_points]
    Y_sub = Y_sub[:num_points]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)

    # Scatter of raw embeddings
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=Y_sub[:, attr_idx],
        cmap="coolwarm",
        s=4,
        alpha=0.5,
    )
    plt.colorbar(label=f"{attr_name} (0/1)")
    plt.title(f"Raw CLIP embeddings – PCA (colored by {attr_name})")
    plt.tight_layout()
    raw_path = os.path.join(output_dir, f"pca_raw_{attr_name.lower()}.png")
    plt.savefig(raw_path, dpi=200)
    plt.close()

    # Select a few trajectories: start from negatives and flow toward positives
    mask_neg = attrs[:, attr_idx] == 0
    neg_indices = torch.where(mask_neg)[0]
    if len(neg_indices) == 0:
        return
    neg_indices = neg_indices[:num_traj]

    traj_points: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        z0 = embeddings[neg_indices].to(device)
        y0 = attrs[neg_indices].to(device)
        y_target = y0.clone()
        y_target[:, attr_idx] = 1

        z_t = z0
        zs = [z_t.cpu()]
        for _ in range(num_steps):
            z_t, _ = integrate_flow(model, z0=z_t, y=y_target, num_steps=1)
            zs.append(z_t.cpu())

        traj_tensor = torch.stack(zs, dim=0)  # (T+1, num_traj, D)

    # Project each step to PCA space using same transform
    for t in range(traj_tensor.shape[0]):
        z_t = traj_tensor[t]
        z_t_np = _to_numpy(z_t)
        traj_points.append(pca.transform(z_t_np))  # (num_traj, 2)

    # Overlay trajectories on top of raw scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=Y_sub[:, attr_idx],
        cmap="coolwarm",
        s=4,
        alpha=0.2,
    )
    for j in range(num_traj):
        xs = [traj_points[t][j, 0] for t in range(len(traj_points))]
        ys = [traj_points[t][j, 1] for t in range(len(traj_points))]
        plt.plot(xs, ys, "-", linewidth=1.0, alpha=0.9)
        plt.scatter(xs[0], ys[0], c="black", s=10)  # start
        plt.scatter(xs[-1], ys[-1], c="green", s=10)  # end

    plt.title(f"Flow trajectories toward {attr_name}=1 in PCA space")
    plt.tight_layout()
    traj_path = os.path.join(output_dir, f"pca_traj_{attr_name.lower()}.png")
    plt.savefig(traj_path, dpi=200)
    plt.close()


def make_vector_field_quiver(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    attr_idx: int,
    attr_name: str,
    num_points: int = 500,
    device: torch.device = torch.device("cpu"),
    output_dir: str = "notebooks/figures",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Subsample points for quiver
    X_np = _to_numpy(embeddings)
    Y_np = _to_numpy(attrs)
    X_sub, Y_sub = sk_shuffle(X_np, Y_np, random_state=1)
    X_sub = X_sub[:num_points]
    Y_sub = Y_sub[:num_points]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)

    # Compute one-step tangential field and project to 2D
    z = torch.from_numpy(X_sub).to(device)
    y = torch.from_numpy(Y_sub).to(device)
    y_target = y.clone()
    y_target[:, attr_idx] = 1

    with torch.no_grad():
        z_next, v_tan = tangent_sphere_step(model.to(device), z, y_target)

    v_np = _to_numpy(v_tan)
    v_pca = v_np @ pca.components_.T  # (num_points, 2)

    # Scale arrows for visualization
    scale = np.percentile(np.linalg.norm(v_pca, axis=1), 95)
    if scale == 0:
        scale = 1.0
    v_pca_scaled = v_pca / scale

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=Y_sub[:, attr_idx],
        cmap="coolwarm",
        s=8,
        alpha=0.6,
    )
    plt.quiver(
        X_pca[:, 0],
        X_pca[:, 1],
        v_pca_scaled[:, 0],
        v_pca_scaled[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        color="black",
        alpha=0.8,
    )
    plt.title(f"Projected vector field toward {attr_name}=1 in PCA space")
    plt.tight_layout()
    quiver_path = os.path.join(output_dir, f"pca_quiver_{attr_name.lower()}.png")
    plt.savefig(quiver_path, dpi=200)
    plt.close()


def make_retrieval_grid(
    model: ConditionalVectorField,
    all_embeddings: torch.Tensor,
    all_filenames: List[str],
    attrs: torch.Tensor,
    attr_idx: int,
    attr_name: str,
    celeba_root: str = "data/celeba",
    num_examples: int = 4,
    num_steps: int = 6,
    device: torch.device = torch.device("cpu"),
    output_dir: str = "notebooks/figures",
) -> None:
    """
    For a few negative examples (attr=0), flow toward attr=1 and at each step
    show the nearest real neighbor image from CelebA.
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # Normalize once for similarity
    all_emb_norm = torch.nn.functional.normalize(all_embeddings.to(device), dim=-1)

    # Choose negative examples
    mask_neg = attrs[:, attr_idx] == 0
    neg_indices = torch.where(mask_neg)[0]
    if len(neg_indices) == 0:
        return
    neg_indices = neg_indices[:num_examples]

    model.eval()
    fig, axes = plt.subplots(
        nrows=num_examples,
        ncols=num_steps + 1,
        figsize=(2 * (num_steps + 1), 2 * num_examples),
    )

    for row, idx in enumerate(neg_indices):
        z0 = all_embeddings[idx : idx + 1].to(device)
        y0 = attrs[idx : idx + 1].to(device)
        y_target = y0.clone()
        y_target[:, attr_idx] = 1

        z_t = z0
        for col in range(num_steps + 1):
            # Find nearest neighbor in current embedding space
            z_norm = torch.nn.functional.normalize(z_t, dim=-1)
            sims = torch.matmul(z_norm, all_emb_norm.t()).squeeze(0)
            nn_idx = int(torch.argmax(sims).item())
            img_fname = all_filenames[nn_idx]
            img_path = os.path.join(celeba_root, "img_align_celeba", img_fname)
            img = Image.open(img_path).convert("RGB")

            ax = axes[row, col] if num_examples > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"step {col}", fontsize=8)

            # Take one flow step for next column
            if col < num_steps:
                z_t, _ = integrate_flow(
                    model,
                    z0=z_t,
                    y=y_target,
                    num_steps=1,
                )

    plt.suptitle(f"Nearest neighbors along flow toward {attr_name}=1", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_path = os.path.join(output_dir, f"retrieval_flow_{attr_name.lower()}.png")
    plt.savefig(grid_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative visualizations for FCLF flows.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.join("checkpoints", "default", "fclf_last.pt"),
        help="Checkpoint to visualize.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with precomputed embeddings.",
    )
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="data/celeba",
        help="Path to CelebA root for retrieving nearest-neighbor images.",
    )
    parser.add_argument(
        "--attr_name",
        type=str,
        default="Smiling",
        help="Attribute to visualize.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of flow steps for trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.checkpoint_path
    embedding_dir = args.embedding_dir
    celeba_root = args.celeba_root
    attr_name = args.attr_name
    num_steps = args.num_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split, test_split = load_embeddings_splits(embedding_dir)
    target_attrs = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
    attr_indices = get_attribute_indices(train_split.attr_names, target_attrs)

    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)
    test_attrs = select_attributes(test_split, attr_indices)

    fallback_cfg = FCLFConfig()
    model = load_trained_model(ckpt_path, cfg_fallback=fallback_cfg).to(device)

    # Use validation embeddings for PCA-based visualizations
    embeddings_val = val_split.embeddings
    attrs_val = val_attrs

    # All embeddings for retrieval
    all_embeddings = torch.cat(
        [train_split.embeddings, val_split.embeddings, test_split.embeddings], dim=0
    )
    all_filenames: List[str] = (
        list(train_split.filenames)
        + list(val_split.filenames)
        + list(test_split.filenames)
    )

    # Focus on a single attribute (e.g., Smiling) for the main visualizations
    if attr_name not in target_attrs:
        raise ValueError(
            f"Attribute '{attr_name}' not in target set {target_attrs}. "
            "Update scripts/visualize_flows.py to include it."
        )
    attr_idx = target_attrs.index(attr_name)

    make_pca_scatter_with_trajectories(
        model=model,
        embeddings=embeddings_val,
        attrs=attrs_val,
        attr_idx=attr_idx,
        attr_name=attr_name,
        num_points=3000,
        num_traj=50,
        num_steps=num_steps,
        device=device,
    )

    make_vector_field_quiver(
        model=model,
        embeddings=embeddings_val,
        attrs=attrs_val,
        attr_idx=attr_idx,
        attr_name=attr_name,
        num_points=500,
        device=device,
    )

    make_retrieval_grid(
        model=model,
        all_embeddings=all_embeddings,
        all_filenames=all_filenames,
        attrs=torch.cat([train_attrs, val_attrs, test_attrs], dim=0),
        attr_idx=attr_idx,
        attr_name=attr_name,
        celeba_root=celeba_root,
        num_examples=4,
        num_steps=min(6, num_steps),
        device=device,
    )


if __name__ == "__main__":
    main()


