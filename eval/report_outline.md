# Report / Presentation Outline (expA+)

## Figures & Tables
- **Quantitative table**: raw vs flowed linear-probe accuracy/F1 (from `scripts/eval_fclf_metrics.py`).
- **Clustering/NN metrics**: silhouette, Davies-Bouldin, purity, NN purity (same script).
- **Trajectory + entanglement curves**: copy the block emitted by the updated script for Smiling, Young, Male, Eyeglasses, Mustache.
- **Visualization set per attribute** (generated via `scripts/visualize_flows.py --output_dir eval/expA ...`):
  - PCA scatter (raw) & PCA + trajectories.
  - Vector-field quiver (with new subsampled arrows).
  - Retrieval grid.

## Narrative Points
1. **Method recap**: function-conditioned vector field on CLIP embeddings, optimized with contrastive + manifold regularizers.
2. **Stability**: reference multi-step shift table (`scripts/eval_fclf_flows.py`).
3. **Effectiveness**: highlight 100% linear probes (expA) vs smoother-but-strong results from soft config (once trained).
4. **Controllability**: discuss trajectory curves & entanglement diagnostics (target attribute increases, others stay mostly flat).
5. **Qualitative evidence**: describe PCA trajectories + retrieval sequences showing semantic edits.
6. **Ablation insight**: compare soft model vs no-identity config (once trained) to show why manifold constraints matter.
7. **Limitations / future work**: mention attribute entanglement where observed (e.g., Smiling ↔ Male), possible per-attribute vector fields.

## Commands to regenerate assets
```bash
# Soft model training (run on VM with torch installed)
python scripts/train_fclf.py --config configs/fclf_config_soft.yaml --output_dir checkpoints/exp_soft

# Ablation model
python scripts/train_fclf.py --config configs/fclf_config_no_identity.yaml --output_dir checkpoints/exp_ablation

# Metrics + entanglement
python scripts/eval_fclf_metrics.py --checkpoint_path checkpoints/exp_soft/fclf_last.pt --num_steps_flow 12 > eval/exp_soft_metrics.txt

# Stability
python scripts/eval_fclf_flows.py --checkpoint_path checkpoints/exp_soft/fclf_last.pt --num_steps 15 > eval/exp_soft_stability.txt

# Visualizations (multi-attribute)
python scripts/visualize_flows.py --checkpoint_path checkpoints/exp_soft/fclf_last.pt \\
  --attr_names Smiling Young Mustache --num_steps 12 --output_dir eval/exp_soft
```

Use the expA outputs already in `eval/expA/` plus the new exp_soft/exp_ablation results to populate the final slides/report.

