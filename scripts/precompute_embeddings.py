import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


try:
    import clip  # type: ignore
except ImportError as e:  # pragma: no cover - import-time guard
    clip = None


ATTR_FILE = "list_attr_celeba.txt"
PARTITION_FILE = "list_eval_partition.txt"
IMG_DIR = "img_align_celeba"


@dataclass
class CelebAEntry:
    filename: str
    attrs: torch.Tensor  # (num_attrs,)
    split: int  # 0=train, 1=val, 2=test


class CelebAImageDataset(Dataset):
    """
    Minimal CelebA wrapper that reads from the official text files and image folder.

    This keeps us independent of `torchvision.datasets.CelebA` and works with a
    manually downloaded dataset in `data/celeba`.
    """

    def __init__(self, root: str, entries: List[CelebAEntry], preprocess):
        self.root = root
        self.entries = entries
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = os.path.join(self.root, IMG_DIR, entry.filename)
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        return img, entry.attrs, entry.filename


def _parse_attr_file(attr_path: str) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """
    Parse `list_attr_celeba.txt`.

    Returns:
        attr_names: list of attribute names (length 40)
        attr_dict:  mapping from filename -> tensor of shape (40,) with {0,1} values
    """
    with open(attr_path, "r") as f:
        lines = f.readlines()

    # First line: number of images
    # Second line: attribute names
    header = lines[1].strip().split()
    attr_names = header

    attr_dict: Dict[str, torch.Tensor] = {}
    for line in lines[2:]:
        parts = line.strip().split()
        if not parts:
            continue
        filename = parts[0]
        # Original labels are -1 or 1
        values = torch.tensor([1 if int(x) == 1 else 0 for x in parts[1:]], dtype=torch.int8)
        attr_dict[filename] = values

    return attr_names, attr_dict


def _parse_partition_file(partition_path: str) -> Dict[str, int]:
    """
    Parse `list_eval_partition.txt` if available.

    Returns:
        mapping from filename -> split_id (0=train, 1=val, 2=test)
    """
    split_map: Dict[str, int] = {}
    with open(partition_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            filename, split_str = parts
            split_map[filename] = int(split_str)
    return split_map


def _build_entries(
    celeba_root: str,
) -> Tuple[List[str], List[CelebAEntry]]:
    attr_path = os.path.join(celeba_root, ATTR_FILE)
    if not os.path.exists(attr_path):
        raise FileNotFoundError(
            f"Could not find {ATTR_FILE} at {attr_path}. "
            "Make sure you downloaded CelebA correctly."
        )

    attr_names, attr_dict = _parse_attr_file(attr_path)

    partition_path = os.path.join(celeba_root, PARTITION_FILE)
    if os.path.exists(partition_path):
        split_map = _parse_partition_file(partition_path)
    else:
        # Fallback: deterministic random split matching official ratios
        print(
            f"[WARN] {PARTITION_FILE} not found at {partition_path}. "
            "Creating a deterministic random train/val/test split."
        )
        rng = torch.Generator().manual_seed(0)
        filenames = sorted(attr_dict.keys())
        perm = torch.randperm(len(filenames), generator=rng).tolist()
        n = len(filenames)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        split_map = {}
        for i, idx in enumerate(perm):
            fname = filenames[idx]
            if i < n_train:
                split_map[fname] = 0
            elif i < n_train + n_val:
                split_map[fname] = 1
            else:
                split_map[fname] = 2

    entries: List[CelebAEntry] = []
    for filename, attrs in attr_dict.items():
        split_id = split_map.get(filename, 0)
        entries.append(CelebAEntry(filename=filename, attrs=attrs, split=split_id))

    return attr_names, entries


def _load_clip_model(device: torch.device):
    if clip is None:
        raise ImportError(
            "The `clip` package is not installed. "
            "Install it with `pip install git+https://github.com/openai/CLIP.git`."
        )
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def _compute_embeddings_for_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    all_embeds: List[torch.Tensor] = []
    all_attrs: List[torch.Tensor] = []
    all_fnames: List[str] = []

    for imgs, attrs, fnames in dataloader:
        imgs = imgs.to(device)
        feats = model.encode_image(imgs)
        feats = nn.functional.normalize(feats, dim=-1)
        all_embeds.append(feats.cpu())
        all_attrs.append(attrs.clone().cpu())
        all_fnames.extend(list(fnames))

    embeddings = torch.cat(all_embeds, dim=0)
    attributes = torch.cat(all_attrs, dim=0)
    return embeddings, attributes, all_fnames


def precompute_embeddings(
    celeba_root: str,
    output_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
    device: str = "cuda",
) -> None:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train_embeddings.pt")
    val_out = os.path.join(output_dir, "val_embeddings.pt")
    test_out = os.path.join(output_dir, "test_embeddings.pt")

    if all(os.path.exists(p) for p in [train_out, val_out, test_out]):
        print("[INFO] All embedding files already exist. Skipping computation.")
        return

    print(f"[INFO] Building CelebA index from {celeba_root}...")
    attr_names, entries = _build_entries(celeba_root)

    print("[INFO] Loading CLIP ViT-B/32 model...")
    model, preprocess = _load_clip_model(device_t)

    # Split entries
    split_to_entries: Dict[int, List[CelebAEntry]] = {0: [], 1: [], 2: []}
    for e in entries:
        split_to_entries[e.split].append(e)

    split_names = {0: "train", 1: "val", 2: "test"}
    split_out_paths = {0: train_out, 1: val_out, 2: test_out}

    for split_id in [0, 1, 2]:
        split_entries = split_to_entries[split_id]
        if not split_entries:
            print(f"[WARN] No entries for split {split_names[split_id]} (id={split_id}). Skipping.")
            continue

        out_path = split_out_paths[split_id]
        if os.path.exists(out_path):
            print(f"[INFO] {out_path} already exists. Skipping {split_names[split_id]} split.")
            continue

        print(f"[INFO] Computing embeddings for {split_names[split_id]} split "
              f"({len(split_entries)} images)...")

        dataset = CelebAImageDataset(celeba_root, split_entries, preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        embeds, attrs, fnames = _compute_embeddings_for_split(model, dataloader, device_t)

        torch.save(
            {
                "embeddings": embeds,          # (N, 512)
                "attributes": attrs,           # (N, 40) int8 {0,1}
                "filenames": fnames,           # list[str]
                "attr_names": attr_names,      # list[str]
                "split": split_names[split_id],
            },
            out_path,
        )
        print(f"[INFO] Saved {split_names[split_id]} embeddings to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for CelebA.")
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="data/celeba",
        help="Path to CelebA root directory containing img_align_celeba/ and list_attr_celeba.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/embeddings",
        help="Directory to store train/val/test embedding .pt files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 is safest on many setups)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device: 'cuda' or 'cpu'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_embeddings(
        celeba_root=args.celeba_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


