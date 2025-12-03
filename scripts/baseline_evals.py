import os
from typing import List, Tuple

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
) -> None:
    """
    Train simple linear classifiers on raw CLIP embeddings for selected attributes.

    Prints accuracy and F1 scores on the validation split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train, val, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train.attr_names, target_attrs)

    X_train = _to_numpy(train.embeddings)
    X_val = _to_numpy(val.embeddings)
    Y_train = select_attributes(train, attr_indices)
    Y_val = select_attributes(val, attr_indices)

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

        print(
            f"{attr_name:10s} | val_acc = {acc: .4f} | val_f1 = {f1: .4f} | "
            f"pos_rate = {pos_rate: .3f}"
        )


def plot_pca_umap(
    embedding_dir: str,
    attr_name: str,
    max_points: int = 10000,
    output_dir: str = "notebooks/figures",
) -> None:
    """
    Create PCA (and optionally UMAP) 2D plots of raw embeddings colored by a single attribute.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    plt.title(f"Raw CLIP Embeddings PCA – colored by {attr_name}")
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"raw_clip_pca_{attr_name.lower()}.png")
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
        plt.title(f"Raw CLIP Embeddings UMAP – colored by {attr_name}")
        plt.tight_layout()
        umap_path = os.path.join(output_dir, f"raw_clip_umap_{attr_name.lower()}.png")
        plt.savefig(umap_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved UMAP plot to {umap_path}")
    else:
        print("[WARN] `umap-learn` is not installed; skipping UMAP plot.")


if __name__ == "__main__":
    EMB_DIR = "data/embeddings"
    TARGET_ATTRS = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]

    run_linear_probes(EMB_DIR, TARGET_ATTRS)
    plot_pca_umap(EMB_DIR, attr_name="Smiling")


