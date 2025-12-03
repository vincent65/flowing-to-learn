import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class EmbeddingSplit:
    """
    Container for a single split of precomputed embeddings.
    """

    embeddings: torch.Tensor  # (N, D), float32, L2-normalized
    attributes: torch.Tensor  # (N, A), int8 {0,1}
    filenames: List[str]      # length N
    attr_names: List[str]     # length A
    split: str                # "train" | "val" | "test"


def _load_split(path: str) -> EmbeddingSplit:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding file {path} not found. "
            "Run scripts/precompute_embeddings.py first."
        )
    data: Dict = torch.load(path, map_location="cpu")

    return EmbeddingSplit(
        embeddings=data["embeddings"],
        attributes=data["attributes"],
        filenames=list(data["filenames"]),
        attr_names=list(data["attr_names"]),
        split=str(data.get("split", "unknown")),
    )


def load_embeddings_splits(
    embedding_dir: str,
) -> Tuple[EmbeddingSplit, EmbeddingSplit, EmbeddingSplit]:
    """
    Convenience loader for train/val/test splits from a directory.
    """
    train = _load_split(os.path.join(embedding_dir, "train_embeddings.pt"))
    val = _load_split(os.path.join(embedding_dir, "val_embeddings.pt"))
    test = _load_split(os.path.join(embedding_dir, "test_embeddings.pt"))
    return train, val, test


def get_attribute_indices(
    attr_names: List[str],
    target_attrs: List[str],
) -> Dict[str, int]:
    """
    Map desired attribute names to their column indices.
    """
    name_to_idx = {name: i for i, name in enumerate(attr_names)}
    missing = [a for a in target_attrs if a not in name_to_idx]
    if missing:
        raise ValueError(
            f"Attributes {missing} not found in CelebA attr_names. "
            f"Available: {attr_names}"
        )
    return {a: name_to_idx[a] for a in target_attrs}


def select_attributes(
    split: EmbeddingSplit,
    attr_indices: Dict[str, int],
) -> torch.Tensor:
    """
    Extract a subset of attributes as a (N, K) tensor of {0,1}.
    """
    idxs = [attr_indices[a] for a in attr_indices.keys()]
    return split.attributes[:, idxs].to(torch.int64)



