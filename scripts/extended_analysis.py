#!/usr/bin/env python3
"""
Extended analysis for differentiating models with saturated accuracy.

When all models achieve ~100% accuracy on the primary task, use:
1. Training efficiency (epochs to converge)
2. Embedding quality (cluster separation, neighborhood purity)
3. Flow smoothness (trajectory statistics)
4. Multi-attribute generalization
5. Robustness (fewer inference steps)
6. Computational cost

Usage:
    python scripts/extended_analysis.py --results_dir eval/param_search
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    get_attribute_indices,
    load_embeddings_splits,
    select_attributes,
)
from fclf_model import FCLFConfig, ConditionalVectorField, integrate_flow

# Publication figure settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
})

TARGET_ATTRS = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]


def load_model_from_checkpoint(ckpt_path: str) -> ConditionalVectorField:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    
    cfg = FCLFConfig(
        embedding_dim=cfg_dict.get("embedding_dim", 512),
        num_attributes=cfg_dict.get("num_attributes", 5),
        hidden_dim=cfg_dict.get("hidden_dim", 256),
        projection_radius=cfg_dict.get("projection_radius", 1.0),
        alpha=cfg_dict.get("alpha", 0.1),
    )
    
    model = ConditionalVectorField(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def compute_flowed_embeddings_with_trajectory(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    num_steps: int,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute flowed embeddings and trajectory statistics.
    """
    model = model.to(device)
    model.eval()
    
    all_flowed = []
    total_path_length = 0.0
    total_samples = 0
    velocity_norms = []
    
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            z_batch = embeddings[i:i+batch_size].to(device)
            y_batch = attrs[i:i+batch_size].to(device)
            
            # Track trajectory
            z_t = z_batch
            batch_path_length = 0.0
            
            for step in range(num_steps):
                z_next, _ = integrate_flow(model, z_t, y_batch, num_steps=1)
                
                # Path length
                step_dist = torch.norm(z_next - z_t, dim=-1).mean().item()
                batch_path_length += step_dist
                
                # Velocity
                v_norm = step_dist  # Since we use step size implicitly
                velocity_norms.append(v_norm)
                
                z_t = z_next
            
            all_flowed.append(z_t.cpu())
            total_path_length += batch_path_length * z_batch.shape[0]
            total_samples += z_batch.shape[0]
    
    flowed = torch.cat(all_flowed, dim=0)
    
    trajectory_stats = {
        "mean_path_length": total_path_length / total_samples,
        "mean_velocity": np.mean(velocity_norms),
        "velocity_std": np.std(velocity_norms),
    }
    
    return flowed, trajectory_stats


def compute_embedding_quality_metrics(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute embedding quality metrics for a binary classification task.
    """
    y = y.astype(int)
    
    # Cluster metrics using true labels
    if len(np.unique(y)) >= 2:
        silhouette = silhouette_score(X, y)
        davies_bouldin = davies_bouldin_score(X, y)
    else:
        silhouette = 0.0
        davies_bouldin = 0.0
    
    # Class centroids and distances
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    if len(class_0) > 0 and len(class_1) > 0:
        centroid_0 = class_0.mean(axis=0)
        centroid_1 = class_1.mean(axis=0)
        
        # Inter-class distance (between centroids)
        inter_class_dist = np.linalg.norm(centroid_0 - centroid_1)
        
        # Intra-class distance (average distance to centroid)
        intra_0 = np.linalg.norm(class_0 - centroid_0, axis=1).mean()
        intra_1 = np.linalg.norm(class_1 - centroid_1, axis=1).mean()
        intra_class_dist = (intra_0 + intra_1) / 2
        
        # Separation ratio (higher is better)
        separation_ratio = inter_class_dist / (intra_class_dist + 1e-8)
    else:
        inter_class_dist = 0.0
        intra_class_dist = 0.0
        separation_ratio = 0.0
    
    # Neighborhood purity (k-NN same-class ratio)
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]  # Exclude self
    purity = (neighbor_labels == y[:, None]).mean()
    
    return {
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "inter_class_dist": float(inter_class_dist),
        "intra_class_dist": float(intra_class_dist),
        "separation_ratio": float(separation_ratio),
        "neighborhood_purity": float(purity),
    }


def compute_convergence_epoch(loss_log_path: str, threshold: float = 0.01) -> int:
    """
    Find the epoch where loss stabilizes (change < threshold).
    """
    if not os.path.exists(loss_log_path):
        return -1
    
    with open(loss_log_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) < 5:
        return len(rows)
    
    losses = [float(row.get("total_loss", 0)) for row in rows]
    
    # Find first epoch where rolling average change is below threshold
    window = 5
    for i in range(window, len(losses)):
        recent_avg = np.mean(losses[i-window:i])
        prev_avg = np.mean(losses[i-window-1:i-1]) if i > window else losses[0]
        
        if abs(recent_avg - prev_avg) / (prev_avg + 1e-8) < threshold:
            return i
    
    return len(losses)


def evaluate_multi_attribute(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    attr_names: List[str],
    num_steps: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate linear probe accuracy on all attributes.
    """
    flowed, _ = compute_flowed_embeddings_with_trajectory(
        model, embeddings, attrs, num_steps, device
    )
    X = flowed.numpy()
    
    results = {}
    for i, attr_name in enumerate(attr_names):
        y = attrs[:, i].numpy()
        
        if len(np.unique(y)) < 2:
            results[attr_name] = {"accuracy": 0.0, "auc": 0.0}
            continue
        
        # Train/test split (use first 80% for train)
        n_train = int(0.8 * len(y))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        if len(np.unique(y_test)) >= 2:
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        else:
            auc = 0.0
        
        results[attr_name] = {"accuracy": float(acc), "auc": float(auc)}
    
    return results


def evaluate_robustness(
    model: ConditionalVectorField,
    embeddings: torch.Tensor,
    attrs: torch.Tensor,
    y_true: np.ndarray,
    max_steps: int,
    device: torch.device,
) -> Dict[int, float]:
    """
    Evaluate accuracy at different numbers of flow steps.
    """
    step_counts = [0, 1, 2, 5, max_steps // 2, max_steps]
    step_counts = sorted(set(s for s in step_counts if s <= max_steps))
    
    results = {}
    
    for n_steps in step_counts:
        if n_steps == 0:
            X = embeddings.numpy()
        else:
            flowed, _ = compute_flowed_embeddings_with_trajectory(
                model, embeddings, attrs, n_steps, device
            )
            X = flowed.numpy()
        
        # Train probe on 80%
        n_train = int(0.8 * len(y_true))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y_true[:n_train], y_true[n_train:]
        
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        results[n_steps] = float(acc)
    
    return results


@dataclass
class ExtendedMetrics:
    run_id: str
    # Original metrics
    val_acc: float
    val_auc: float
    # Training efficiency
    convergence_epoch: int
    final_loss: float
    # Embedding quality
    silhouette: float
    davies_bouldin: float
    separation_ratio: float
    neighborhood_purity: float
    # Flow characteristics
    mean_path_length: float
    mean_velocity: float
    velocity_std: float
    # Multi-attribute
    multi_attr_mean_acc: float
    # Robustness
    robustness_scores: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "val_acc": self.val_acc,
            "val_auc": self.val_auc,
            "convergence_epoch": self.convergence_epoch,
            "final_loss": self.final_loss,
            "silhouette": self.silhouette,
            "davies_bouldin": self.davies_bouldin,
            "separation_ratio": self.separation_ratio,
            "neighborhood_purity": self.neighborhood_purity,
            "mean_path_length": self.mean_path_length,
            "mean_velocity": self.mean_velocity,
            "velocity_std": self.velocity_std,
            "multi_attr_mean_acc": self.multi_attr_mean_acc,
            "robustness_scores": self.robustness_scores,
        }


def run_extended_analysis(
    results_json_path: str,
    embedding_dir: str,
    output_dir: str,
    max_models: int = 50,
) -> List[ExtendedMetrics]:
    """
    Run extended analysis on all models in results.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_json_path, "r") as f:
        results = json.load(f)
    
    # Load embeddings
    print("[Extended Analysis] Loading embeddings...")
    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train_split.attr_names, TARGET_ATTRS)
    val_attrs = select_attributes(val_split, attr_indices)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Extended Analysis] Using device: {device}")
    
    # Use subset for faster analysis
    max_samples = 5000
    subset_idx = torch.randperm(val_split.embeddings.shape[0])[:max_samples]
    val_embeddings_subset = val_split.embeddings[subset_idx]
    val_attrs_subset = val_attrs[subset_idx]
    y_smiling = val_attrs_subset[:, 0].numpy()  # Smiling is index 0
    
    extended_results: List[ExtendedMetrics] = []
    
    for i, r in enumerate(results[:max_models]):
        run_id = r["run_id"]
        print(f"\n[Extended Analysis] Processing {i+1}/{min(len(results), max_models)}: {run_id}")
        
        ckpt_path = r["checkpoint_path"]
        loss_log_path = r.get("loss_log_path", "")
        num_steps = r.get("num_flow_steps", 10)
        
        # Handle relative paths from VM
        if not os.path.isabs(ckpt_path):
            # Try to find it relative to results_json
            base_dir = Path(results_json_path).parent
            ckpt_path_try = base_dir / ckpt_path.replace("eval/param_search/", "")
            if ckpt_path_try.exists():
                ckpt_path = str(ckpt_path_try)
            else:
                print(f"  [WARN] Checkpoint not found: {ckpt_path}")
                continue
        
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path}")
            continue
        
        # Load model
        try:
            model = load_model_from_checkpoint(ckpt_path)
        except Exception as e:
            print(f"  [WARN] Failed to load model: {e}")
            continue
        
        # Convergence epoch
        if loss_log_path and not os.path.isabs(loss_log_path):
            base_dir = Path(results_json_path).parent
            loss_log_try = base_dir / loss_log_path.replace("eval/param_search/", "")
            if loss_log_try.exists():
                loss_log_path = str(loss_log_try)
        
        convergence_epoch = compute_convergence_epoch(loss_log_path)
        
        # Get final loss
        final_loss = 0.0
        if os.path.exists(loss_log_path):
            with open(loss_log_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                final_loss = float(rows[-1].get("total_loss", 0))
        
        # Skip if K=0 (no flow to analyze)
        if num_steps == 0:
            extended_results.append(ExtendedMetrics(
                run_id=run_id,
                val_acc=r["val_acc"],
                val_auc=r["val_auc"],
                convergence_epoch=convergence_epoch,
                final_loss=final_loss,
                silhouette=0.0,
                davies_bouldin=0.0,
                separation_ratio=0.0,
                neighborhood_purity=0.0,
                mean_path_length=0.0,
                mean_velocity=0.0,
                velocity_std=0.0,
                multi_attr_mean_acc=r["val_acc"],
                robustness_scores={0: r["val_acc"]},
            ))
            continue
        
        # Compute flowed embeddings with trajectory stats
        print("  Computing flow trajectory...")
        flowed, traj_stats = compute_flowed_embeddings_with_trajectory(
            model, val_embeddings_subset, val_attrs_subset, num_steps, device
        )
        
        # Embedding quality
        print("  Computing embedding quality metrics...")
        X_flowed = flowed.numpy()
        quality_metrics = compute_embedding_quality_metrics(X_flowed, y_smiling)
        
        # Multi-attribute
        print("  Evaluating multi-attribute performance...")
        multi_attr = evaluate_multi_attribute(
            model, val_embeddings_subset, val_attrs_subset, TARGET_ATTRS, num_steps, device
        )
        multi_attr_mean = np.mean([v["accuracy"] for v in multi_attr.values()])
        
        # Robustness
        print("  Evaluating robustness...")
        robustness = evaluate_robustness(
            model, val_embeddings_subset, val_attrs_subset, y_smiling, num_steps, device
        )
        
        metrics = ExtendedMetrics(
            run_id=run_id,
            val_acc=r["val_acc"],
            val_auc=r["val_auc"],
            convergence_epoch=convergence_epoch,
            final_loss=final_loss,
            silhouette=quality_metrics["silhouette"],
            davies_bouldin=quality_metrics["davies_bouldin"],
            separation_ratio=quality_metrics["separation_ratio"],
            neighborhood_purity=quality_metrics["neighborhood_purity"],
            mean_path_length=traj_stats["mean_path_length"],
            mean_velocity=traj_stats["mean_velocity"],
            velocity_std=traj_stats["velocity_std"],
            multi_attr_mean_acc=multi_attr_mean,
            robustness_scores=robustness,
        )
        
        extended_results.append(metrics)
        
        print(f"  silhouette={metrics.silhouette:.4f}, separation={metrics.separation_ratio:.4f}")
        print(f"  path_length={metrics.mean_path_length:.4f}, multi_attr_acc={metrics.multi_attr_mean_acc:.4f}")
    
    # Save results
    output_json = output_dir / "extended_metrics.json"
    with open(output_json, "w") as f:
        json.dump([m.to_dict() for m in extended_results], f, indent=2)
    print(f"\n[Extended Analysis] Saved metrics to {output_json}")
    
    # Generate figures
    generate_extended_figures(extended_results, output_dir)
    
    return extended_results


def generate_extended_figures(metrics: List[ExtendedMetrics], output_dir: Path) -> None:
    """Generate comparison figures for extended metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out models with no flow
    flow_metrics = [m for m in metrics if m.mean_path_length > 0]
    
    if not flow_metrics:
        print("[Figures] No flow metrics to plot")
        return
    
    # Sort by separation ratio (descending)
    sorted_metrics = sorted(flow_metrics, key=lambda m: m.separation_ratio, reverse=True)
    
    # Figure 1: Multi-metric comparison (spider/radar chart alternative - bar groups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Take top 10 models for clarity
    top_n = min(10, len(sorted_metrics))
    top_metrics = sorted_metrics[:top_n]
    names = [m.run_id for m in top_metrics]
    
    # Subplot 1: Silhouette Score
    ax = axes[0, 0]
    values = [m.silhouette for m in top_metrics]
    ax.barh(range(len(names)), values, color='#2E86AB')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Silhouette Score")
    ax.set_title("Cluster Quality (higher = better)")
    ax.invert_yaxis()
    
    # Subplot 2: Separation Ratio
    ax = axes[0, 1]
    values = [m.separation_ratio for m in top_metrics]
    ax.barh(range(len(names)), values, color='#A23B72')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Inter/Intra Class Distance Ratio")
    ax.set_title("Class Separation (higher = better)")
    ax.invert_yaxis()
    
    # Subplot 3: Neighborhood Purity
    ax = axes[0, 2]
    values = [m.neighborhood_purity for m in top_metrics]
    ax.barh(range(len(names)), values, color='#F18F01')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("k-NN Same-Class Ratio")
    ax.set_title("Neighborhood Purity (higher = better)")
    ax.invert_yaxis()
    
    # Subplot 4: Path Length
    ax = axes[1, 0]
    values = [m.mean_path_length for m in top_metrics]
    ax.barh(range(len(names)), values, color='#C73E1D')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Trajectory Length")
    ax.set_title("Flow Path Length")
    ax.invert_yaxis()
    
    # Subplot 5: Multi-attribute Mean Accuracy
    ax = axes[1, 1]
    values = [m.multi_attr_mean_acc for m in top_metrics]
    ax.barh(range(len(names)), values, color='#28A745')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Accuracy (5 attributes)")
    ax.set_title("Multi-Attribute Generalization")
    ax.invert_yaxis()
    
    # Subplot 6: Convergence Epoch
    ax = axes[1, 2]
    values = [m.convergence_epoch for m in top_metrics]
    ax.barh(range(len(names)), values, color='#6C757D')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Epochs to Converge")
    ax.set_title("Training Efficiency (lower = better)")
    ax.invert_yaxis()
    
    fig.suptitle("Extended Model Comparison (Top 10 by Separation Ratio)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    path = output_dir / "extended_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figures] Saved: {path}")
    
    # Figure 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metric_names = ["silhouette", "separation_ratio", "neighborhood_purity", 
                    "mean_path_length", "multi_attr_mean_acc", "convergence_epoch"]
    
    data = np.array([
        [getattr(m, name) for m in flow_metrics]
        for name in metric_names
    ])
    
    corr = np.corrcoef(data)
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(metric_names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in metric_names], fontsize=9)
    ax.set_yticklabels([n.replace("_", "\n") for n in metric_names], fontsize=9)
    
    # Add correlation values
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    path = output_dir / "metric_correlations.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figures] Saved: {path}")
    
    # Figure 3: Robustness curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(flow_metrics))))
    
    for idx, m in enumerate(sorted_metrics[:10]):
        steps = sorted(m.robustness_scores.keys())
        accs = [m.robustness_scores[s] for s in steps]
        ax.plot(steps, accs, 'o-', label=m.run_id, color=colors[idx], linewidth=1.5, markersize=6)
    
    ax.set_xlabel("Number of Flow Steps")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Robustness: Accuracy vs Flow Steps", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    
    fig.tight_layout()
    path = output_dir / "robustness_curves.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Figures] Saved: {path}")
    
    # Figure 4: Summary ranking table
    create_ranking_table(sorted_metrics, output_dir)


def create_ranking_table(metrics: List[ExtendedMetrics], output_dir: Path) -> None:
    """Create a summary ranking table."""
    
    # Compute composite score (normalize and average key metrics)
    def normalize(values, higher_is_better=True):
        values = np.array(values)
        min_v, max_v = values.min(), values.max()
        if max_v - min_v < 1e-8:
            return np.ones_like(values) * 0.5
        normalized = (values - min_v) / (max_v - min_v)
        return normalized if higher_is_better else 1 - normalized
    
    sil_norm = normalize([m.silhouette for m in metrics])
    sep_norm = normalize([m.separation_ratio for m in metrics])
    pur_norm = normalize([m.neighborhood_purity for m in metrics])
    multi_norm = normalize([m.multi_attr_mean_acc for m in metrics])
    conv_norm = normalize([m.convergence_epoch for m in metrics], higher_is_better=False)
    
    composite_scores = (sil_norm + sep_norm + pur_norm + multi_norm + conv_norm) / 5
    
    # Create CSV
    csv_path = output_dir / "model_rankings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "run_id", "composite_score", "val_acc",
            "silhouette", "separation_ratio", "neighborhood_purity",
            "multi_attr_acc", "convergence_epoch"
        ])
        writer.writeheader()
        
        ranked = sorted(zip(metrics, composite_scores), key=lambda x: x[1], reverse=True)
        
        for rank, (m, score) in enumerate(ranked, 1):
            writer.writerow({
                "rank": rank,
                "run_id": m.run_id,
                "composite_score": f"{score:.4f}",
                "val_acc": f"{m.val_acc:.4f}",
                "silhouette": f"{m.silhouette:.4f}",
                "separation_ratio": f"{m.separation_ratio:.4f}",
                "neighborhood_purity": f"{m.neighborhood_purity:.4f}",
                "multi_attr_acc": f"{m.multi_attr_mean_acc:.4f}",
                "convergence_epoch": m.convergence_epoch,
            })
    
    print(f"[Rankings] Saved: {csv_path}")
    
    # Print top 5
    print("\n" + "="*80)
    print("TOP 5 MODELS (by composite score)")
    print("="*80)
    print(f"{'Rank':<5} {'Run ID':<25} {'Composite':<10} {'Silhouette':<12} {'Separation':<12}")
    print("-"*80)
    
    ranked = sorted(zip(metrics, composite_scores), key=lambda x: x[1], reverse=True)
    for rank, (m, score) in enumerate(ranked[:5], 1):
        print(f"{rank:<5} {m.run_id:<25} {score:<10.4f} {m.silhouette:<12.4f} {m.separation_ratio:<12.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended analysis for ablation study")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval/param_search",
        help="Directory containing results.json",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with precomputed embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to results_dir/extended_analysis)",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=50,
        help="Maximum number of models to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    results_json = results_dir / "results.json"
    
    if not results_json.exists():
        print(f"[ERROR] results.json not found at {results_json}")
        return
    
    output_dir = args.output_dir or str(results_dir / "extended_analysis")
    
    run_extended_analysis(
        results_json_path=str(results_json),
        embedding_dir=args.embedding_dir,
        output_dir=output_dir,
        max_models=args.max_models,
    )


if __name__ == "__main__":
    main()

