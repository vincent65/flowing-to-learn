# Learning to Flow by Flowing to Learn

**CS229 Final Project - Fall 2025**

A flow matching approach for learning attribute-aware representations in CLIP embedding space. The model learns a conditional vector field that transforms embeddings on a hypersphere to improve facial attribute classification.

FCLF learns a neural vector field `v(z, y)` that flows CLIP embeddings conditioned on facial attributes:

```
z_{t+1} = normalize(z_t + α · v_tan(z_t, y))
```

Key components:
- **Spherical geometry**: Embeddings constrained to unit hypersphere
- **Conditional flow**: Vector field conditioned on 5 binary attributes
- **Contrastive learning**: Supervised contrastive loss for attribute clustering
- **Regularization**: Curl and divergence penalties for smooth flows

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── fclf_model.py              # Core model (ConditionalVectorField)
├── celeba_embeddings.py       # Data loading utilities
├── configs/
│   └── fclf_config.yaml       # Training configuration
├── scripts/
│   ├── train_fclf.py          # Training script
│   ├── eval_fclf_metrics.py   # Comprehensive evaluation metrics
│   ├── eval_fclf_flows.py     # Flow-specific evaluation
│   ├── baseline_evals.py      # Baseline comparisons (raw CLIP, global direction)
│   ├── hp_search.py           # Hyperparameter search & ablations
│   ├── extended_analysis.py   # Extended analysis (cluster quality, etc.)
│   ├── describe_datasets.py   # Dataset statistics and visualization
│   ├── visualize_flows.py     # Flow trajectory visualization
│   ├── viz_utils.py           # Plotting utilities
│   └── precompute_embeddings.py  # CLIP embedding extraction
├── notebooks/
│   └── cs229.ipynb            # Experimental analysis notebook
└── data/                      # CelebA dataset (not included)
```

## Usage

### 1. Precompute CLIP Embeddings
```bash
python scripts/precompute_embeddings.py --celeba_root data/celeba --output_dir data/embeddings
```

### 2. Train Model
```bash
python scripts/train_fclf.py --config configs/fclf_config.yaml --output_dir checkpoints/run1
```

### 3. Run Ablation Study
```bash
# Full 30-config ablation study
python scripts/hp_search.py --ablation

# View results
python scripts/hp_search.py --summarize

# Run full evaluation on best model
python scripts/hp_search.py --eval-best
```

### 4. Evaluate Model
```bash
python scripts/eval_fclf_metrics.py --checkpoint_path checkpoints/run1/fclf_last.pt
```

### 5. Extended Analysis
```bash
python scripts/extended_analysis.py --results_dir eval/param_search
```

## Model Architecture

```
ConditionalVectorField: v(z, y) : R^512 × R^5 → R^512

Attribute MLP:    5 → 256 (GELU)
Main MLP:         768 → 256 → 256 → 512 (GELU)
Output:           tanh(·) × scale

```

## Configuration

Key hyperparameters in `configs/fclf_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 512 | CLIP embedding dimension |
| `hidden_dim` | 256 | MLP hidden size |
| `num_epochs` | 50 | Training epochs |
| `batch_size` | 512 | Batch size |
| `learning_rate` | 1e-4 | Adam learning rate |
| `alpha` | 0.1 | Flow step size |
| `temperature` | 0.1 | Contrastive temperature (0.5 is optimal) |
| `lambda_contrastive` | 0.7 | Contrastive loss weight |
| `lambda_identity` | 0.3 | Identity loss weight |
| `num_flow_steps` | 10 | Inference flow steps |

## Attributes

The model is trained on 5 CelebA attributes:
- Smiling
- Young
- Male
- Eyeglasses
- Mustache

## License

MIT License

