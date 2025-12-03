import argparse
import os
from typing import List

import numpy as np
import torch

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


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def run_linear_probes(
    embedding_dir: str,
    target_attrs: List[str],
    output_path: Path,
) -> None:
    """
    Train simple linear classifiers on raw CLIP embeddings for selected attributes.
    Stores the summary in output_path.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train, val, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train.attr_names, target_attrs)

    X_train = _to_numpy(train.embeddings)
    X_val = _to_numpy(val.embeddings)
    Y_train = select_attributes(train, attr_indices)
    Y_val = select_attributes(val, attr_indices)

    output_lines = ["=== Linear Probes on Raw CLIP Embeddings ==="]
    print("=== Linear Probes on Raw CLIP Embeddings ===")
    for j, attr_name in enumerate(attr_indices.keys()):
        y_tr = _to_numpy(Y_train[:, j])
        y_va = _to_numpy(Y_val[:, j])

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, y_tr)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_va, y_pred)
        f1 = f1_score(y_va, y_pred)
        pos_rate = y_va.mean()

        line = (
            f"{attr_name:10s} | val_acc = {acc: .4f} | val_f1 = {f1: .4f} | "
            f"pos_rate = {pos_rate: .3f}"
        )
        print(line)
        output_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines))
    print(f"[INFO] Saved baseline metrics to {output_path}")


def plot_pca_umap(
    embedding_dir: str,
    attr_name: str,
    max_points: int,
    output_dir: Path,
) -> None:
    """
    Create PCA (and optionally UMAP) 2D plots of raw embeddings colored by a single attribute.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    try:
        import umap  # type: ignore
    except ImportError:
        umap = None

    train, _, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train.attr_names, [attr_name])
    Y = select_attributes(train, attr_indices)[:, 0]

    X = train.embeddings
    N = X.shape[0]
    if N > max_points:
        idx = torch.randperm(N)[:max_points]
        X = X[idx]
        Y = Y[idx]

    X_np = _to_numpy(X)
    Y_np = _to_numpy(Y)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=Y_np,
        cmap="coolwarm",
        s=4,
        alpha=0.5,
    )
    plt.colorbar(label=f"{attr_name} (0/1)")
    plt.title(f"Raw CLIP embeddings PCA – colored by {attr_name}")
    plt.tight_layout()
    pca_path = output_dir / f"raw_clip_pca_{attr_name.lower()}.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved PCA plot to {pca_path}")

    # Optional UMAP
    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=0)
        X_umap = reducer.fit_transform(X_np)
        plt.figure(figsize=(6, 5))
        plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=Y_np,
            cmap="coolwarm",
            s=4,
            alpha=0.5,
        )
        plt.colorbar(label=f"{attr_name} (0/1)")
        plt.title(f"Raw CLIP embeddings UMAP – colored by {attr_name}")
        plt.tight_layout()
        umap_path = output_dir / f"raw_clip_umap_{attr_name.lower()}.png"
        plt.savefig(umap_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved UMAP plot to {umap_path}")
    else:
        print("[WARN] `umap-learn` is not installed; skipping UMAP plot.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline CLIP embedding evaluations.")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with precomputed embeddings.",
    )
    parser.add_argument(
        "--attr_names",
        type=str,
        nargs="+",
        default=["Smiling", "Young", "Male", "Eyeglasses", "Mustache"],
        help="Attributes to evaluate.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="eval/baseline/linear_probe_metrics.txt",
        help="Path to save linear probe summary.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="eval/baseline",
        help="Directory to store PCA/UMAP plots.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Max number of embeddings to use for PCA/UMAP plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_path = Path(args.metrics_output)
    figure_dir = Path(args.figure_dir)

    run_linear_probes(args.embedding_dir, args.attr_names, metrics_path)
    for attr in args.attr_names:
        plot_pca_umap(
            embedding_dir=args.embedding_dir,
            attr_name=attr,
            max_points=args.max_points,
            output_dir=figure_dir,
        )


