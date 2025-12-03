import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


def compute_flowed_embeddings(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    num_steps: int,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    ds = EmbeddingDataset(embeddings, attrs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval()

    flowed_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for z_batch, y_batch in loader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            zK, _ = integrate_flow(
                model,
                z0=z_batch,
                y=y_batch,
                num_steps=num_steps,
            )
            flowed_chunks.append(zK.cpu())

    return torch.cat(flowed_chunks, dim=0)


def train_linear_probes(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    attr_names: List[str],
) -> Dict[str, "LogisticRegression"]:
    from sklearn.linear_model import LogisticRegression

    models: Dict[str, LogisticRegression] = {}
    num_attrs = Y_train.shape[1]
    for j in range(num_attrs):
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, Y_train[:, j])
        models[attr_names[j]] = clf
    return models


def eval_linear_probes(
    models: Dict[str, "LogisticRegression"],
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    from sklearn.metrics import accuracy_score, f1_score

    results: Dict[str, Tuple[float, float]] = {}
    num_attrs = Y_val.shape[1]
    for j, (attr_name, clf) in enumerate(models.items()):
        y_true = Y_val[:, j]
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results[attr_name] = (acc, f1)
    return results


def clustering_metrics(
    X: np.ndarray,
    y: np.ndarray,
    attr_name: str,
    n_clusters: int = 2,
) -> Tuple[float, float, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)

    # Cluster purity for this binary attribute
    purity_count = 0
    total = len(y)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        majority = np.round(y[mask].mean())
        purity_count += (y[mask] == majority).sum()
    purity = purity_count / total
    return sil, db, purity


def neighborhood_purity(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Exclude self (first neighbor)
    neighbor_labels = y[indices[:, 1:]]  # (N, k)
    same = (neighbor_labels == y[:, None]).mean()
    return float(same)


def attribute_trajectory_curves(
    model: ConditionalVectorField,
    clf: "LogisticRegression",
    X_init: torch.Tensor,
    Y_init: torch.Tensor,
    attr_idx: int,
    num_steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    For a subset of samples with attribute=0, flow towards target=1 and track
    classifier probability p(attr=1) along the trajectory.

    Returns:
        probs: (num_steps+1,) averaged over the subset.
    """
    # Select negatives to edit
    mask = Y_init[:, attr_idx] == 0
    X0 = X_init[mask]
    Y0 = Y_init[mask]

    if X0.shape[0] == 0:
        return np.zeros(num_steps + 1, dtype=np.float32)

    ds = EmbeddingDataset(X0, Y0)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    model.eval()
    traj_probs: List[np.ndarray] = []

    with torch.no_grad():
        for z_batch, y_batch in loader:
            z = z_batch.to(device)
            y = y_batch.to(device)
            # Define target condition: flip this attribute to 1
            y_target = y.clone()
            y_target[:, attr_idx] = 1

            z_t = z
            probs_steps: List[np.ndarray] = []
            # Step 0: before flow
            logits0 = clf.predict_proba(_to_numpy(z_t))[:, 1]
            probs_steps.append(logits0)

            for _ in range(num_steps):
                z_t, _ = integrate_flow(
                    model,
                    z0=z_t,
                    y=y_target,
                    num_steps=1,
                )
                logits = clf.predict_proba(_to_numpy(z_t))[:, 1]
                probs_steps.append(logits)

            # probs_steps: (num_steps+1, B_chunk)
            probs_mat = np.stack(probs_steps, axis=0)
            traj_probs.append(probs_mat.mean(axis=1))  # (num_steps+1,)

    all_traj = np.stack(traj_probs, axis=0)  # (num_chunks, num_steps+1)
    return all_traj.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitative evaluation for FCLF.")
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
        help="Directory with precomputed embeddings.",
    )
    parser.add_argument(
        "--num_steps_flow",
        type=int,
        default=10,
        help="Number of flow steps to apply when computing flowed embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.checkpoint_path
    embedding_dir = args.embedding_dir
    num_steps_flow = args.num_steps_flow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    target_attrs = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
    attr_indices = get_attribute_indices(train_split.attr_names, target_attrs)

    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)

    # Raw embeddings
    X_train_raw = _to_numpy(train_split.embeddings)
    X_val_raw = _to_numpy(val_split.embeddings)
    Y_train = _to_numpy(train_attrs)
    Y_val = _to_numpy(val_attrs)

    # Train linear probes on raw embeddings
    raw_probe_models = train_linear_probes(X_train_raw, Y_train, target_attrs)
    raw_probe_results = eval_linear_probes(raw_probe_models, X_val_raw, Y_val)

    print("=== Linear probes on RAW CLIP embeddings ===")
    for attr_name, (acc, f1) in raw_probe_results.items():
        print(f"{attr_name:10s} | val_acc = {acc: .4f} | val_f1 = {f1: .4f}")

    # Load model and compute flowed embeddings
    fallback_cfg = FCLFConfig()
    model = load_trained_model(ckpt_path, cfg_fallback=fallback_cfg).to(device)
    X_train_flow = compute_flowed_embeddings(
        model=model,
        embeddings=train_split.embeddings,
        attrs=train_attrs,
        num_steps=num_steps_flow,
        device=device,
    )
    X_val_flow = compute_flowed_embeddings(
        model=model,
        embeddings=val_split.embeddings,
        attrs=val_attrs,
        num_steps=num_steps_flow,
        device=device,
    )

    X_train_flow_np = _to_numpy(X_train_flow)
    X_val_flow_np = _to_numpy(X_val_flow)

    # Linear probes on flowed embeddings
    flow_probe_models = train_linear_probes(X_train_flow_np, Y_train, target_attrs)
    flow_probe_results = eval_linear_probes(flow_probe_models, X_val_flow_np, Y_val)

    print("\n=== Linear probes on FCLF-FLOWED embeddings ===")
    for attr_name, (acc, f1) in flow_probe_results.items():
        print(f"{attr_name:10s} | val_acc = {acc: .4f} | val_f1 = {f1: .4f}")

    # Clustering + neighborhood purity on validation split
    from sklearn.utils import shuffle as sk_shuffle

    max_points = 20000
    X_raw_sub, Y_raw_sub = sk_shuffle(X_val_raw, Y_val, random_state=0)
    X_flow_sub, Y_flow_sub = sk_shuffle(X_val_flow_np, Y_val, random_state=0)
    X_raw_sub = X_raw_sub[:max_points]
    Y_raw_sub = Y_raw_sub[:max_points]
    X_flow_sub = X_flow_sub[:max_points]
    Y_flow_sub = Y_flow_sub[:max_points]

    print("\n=== Clustering and neighborhood metrics (VAL subset) ===")
    for j, attr_name in enumerate(target_attrs):
        y_raw_attr = Y_raw_sub[:, j]
        y_flow_attr = Y_flow_sub[:, j]

        sil_raw, db_raw, purity_raw = clustering_metrics(
            X_raw_sub, y_raw_attr, attr_name
        )
        sil_flow, db_flow, purity_flow = clustering_metrics(
            X_flow_sub, y_flow_attr, attr_name
        )
        nn_purity_raw = neighborhood_purity(X_raw_sub, y_raw_attr, k=10)
        nn_purity_flow = neighborhood_purity(X_flow_sub, y_flow_attr, k=10)

        print(
            f"{attr_name:10s} | "
            f"sil_raw={sil_raw: .3f}, sil_flow={sil_flow: .3f} | "
            f"db_raw={db_raw: .3f}, db_flow={db_flow: .3f} | "
            f"purity_raw={purity_raw: .3f}, purity_flow={purity_flow: .3f} | "
            f"NNpur_raw={nn_purity_raw: .3f}, NNpur_flow={nn_purity_flow: .3f}"
        )

    # Attribute trajectory curves using raw linear probes as frozen classifiers
    print("\n=== Attribute trajectory curves (mean p(attr=1) vs step) ===")
    for j, attr_name in enumerate(target_attrs):
        clf = raw_probe_models[attr_name]
        curve = attribute_trajectory_curves(
            model=model,
            clf=clf,
            X_init=train_split.embeddings,
            Y_init=train_attrs,
            attr_idx=j,
            num_steps=num_steps_flow,
            device=device,
        )
        print(f"{attr_name:10s}:", " ".join(f"{p: .3f}" for p in curve))


if __name__ == "__main__":
    main()


