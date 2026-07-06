"""Graph dataset loading for OUGP case studies."""

from __future__ import annotations

import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


PLANETOID_BASE_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")
AMAZON_PHOTO_URL = (
    "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
    "amazon_electronics_photo.npz"
)


@dataclass(frozen=True)
class CitationGraph:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    task_type: str = "multiclass"
    metric_name: str = "accuracy"

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        if self.y.ndim == 2:
            return int(self.y.size(1))
        return int(self.y.max().item() + 1)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def ensure_planetoid_raw(root: Path, name: str) -> Path:
    raw_dir = root / name.lower() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for suffix in PLANETOID_FILES:
        filename = f"ind.{name.lower()}.{suffix}"
        path = raw_dir / filename
        if not path.exists():
            _download(f"{PLANETOID_BASE_URL}/{filename}", path)
    return raw_dir


def _parse_index_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _sample_mask(indices: list[int] | np.ndarray, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _row_normalize(mx: sp.spmatrix) -> sp.csr_matrix:
    rowsum = np.asarray(mx.sum(1)).flatten()
    inv = np.power(rowsum, -1.0, where=rowsum != 0)
    inv[rowsum == 0] = 0.0
    return sp.diags(inv).dot(mx).tocsr()


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_planetoid(root: str | Path, name: str = "cora") -> CitationGraph:
    """Load Cora/CiteSeer/PubMed using the standard Planetoid split."""

    root = Path(root)
    raw_dir = ensure_planetoid_raw(root, name)
    prefix = raw_dir / f"ind.{name.lower()}"

    x, tx, allx, y, ty, ally, graph = [_load_pickle(Path(f"{prefix}.{s}")) for s in PLANETOID_FILES[:-1]]
    test_idx_reorder = np.array(_parse_index_file(Path(f"{prefix}.test.index")), dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    if name.lower() == "citeseer":
        full_range = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(full_range), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(full_range), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _row_normalize(features)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = labels.argmax(axis=1)

    edges = []
    for src, dsts in graph.items():
        for dst in dsts:
            edges.append((src, dst))
            edges.append((dst, src))
    edge_index_np = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
    edge_index = torch.as_tensor(edge_index_np.T, dtype=torch.long)

    n_nodes = labels.shape[0]
    train_mask = _sample_mask(range(len(y)), n_nodes)
    val_mask = _sample_mask(range(len(y), len(y) + 500), n_nodes)
    test_mask = _sample_mask(test_idx_range, n_nodes)

    return CitationGraph(
        x=torch.as_tensor(features.toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _split_masks(num_nodes: int, seed: int, train_ratio: float = 0.10, val_ratio: float = 0.10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    train_end = int(num_nodes * train_ratio)
    val_end = train_end + int(num_nodes * val_ratio)
    train_mask = _sample_mask(perm[:train_end], num_nodes)
    val_mask = _sample_mask(perm[train_end:val_end], num_nodes)
    test_mask = _sample_mask(perm[val_end:], num_nodes)
    return train_mask, val_mask, test_mask


def _index_mask(index: torch.Tensor | np.ndarray, size: int) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        index = index.cpu().numpy()
    return _sample_mask(np.asarray(index, dtype=np.int64).reshape(-1), size)


def _dataset_root(root: Path, family: str, name: str) -> Path:
    if root.name == "planetoid":
        root = root.parent
    return root / family / name / "raw"


def ensure_amazon_photo_raw(root: Path) -> Path:
    raw_dir = _dataset_root(root, "amazon", "photo")
    path = raw_dir / "amazon_electronics_photo.npz"
    if not path.exists():
        _download(AMAZON_PHOTO_URL, path)
    return path


def _adj_from_npz(loader: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    if {"adj_data", "adj_indices", "adj_indptr", "adj_shape"}.issubset(loader.files):
        return sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )
    if "adj_matrix" in loader.files:
        adj = loader["adj_matrix"].item()
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        return adj.tocsr()
    raise KeyError(f"Unsupported Amazon Photo adjacency keys: {loader.files}")


def _features_from_npz(loader: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    if {"attr_data", "attr_indices", "attr_indptr", "attr_shape"}.issubset(loader.files):
        return sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )
    if "attr_matrix" in loader.files:
        attr = loader["attr_matrix"].item()
        if not sp.issparse(attr):
            attr = sp.csr_matrix(attr)
        return attr.tocsr()
    if "features" in loader.files:
        return sp.csr_matrix(loader["features"])
    raise KeyError(f"Unsupported Amazon Photo feature keys: {loader.files}")


def load_amazon_photo(root: str | Path, split_seed: int = 0) -> CitationGraph:
    """Load Amazon Photo from the public gnn-benchmark NPZ file.

    The dataset has no canonical Planetoid split here, so this loader uses a
    fixed 10%/10%/80% train/validation/test split.
    """

    path = ensure_amazon_photo_raw(Path(root))
    with np.load(path, allow_pickle=True) as loader:
        adj = _adj_from_npz(loader)
        features = _features_from_npz(loader)
        labels = loader["labels"].astype(np.int64)

    features = _row_normalize(features)
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()

    coo = adj.tocoo()
    edge_index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    train_mask, val_mask, test_mask = _split_masks(labels.shape[0], split_seed)

    return CitationGraph(
        x=torch.as_tensor(features.toarray(), dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_graph_dataset(root: str | Path, name: str) -> CitationGraph:
    lowered = name.lower()
    if lowered in {"cora", "citeseer", "pubmed"}:
        return load_planetoid(root, lowered)
    if lowered in {"photo", "amazon_photo", "amazon-photo"}:
        return load_amazon_photo(root)
    if lowered in {"ogbn-arxiv", "arxiv"}:
        return load_ogbn_node_property(root, "ogbn-arxiv")
    if lowered in {"ogbn-products", "products"}:
        return load_ogbn_node_property(root, "ogbn-products")
    if lowered in {"ogbn-proteins", "proteins"}:
        return load_ogbn_node_property(root, "ogbn-proteins")
    raise ValueError(f"Unknown dataset {name!r}.")


def _aggregate_edge_features(num_nodes: int, edge_index: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
    features = np.zeros((num_nodes, edge_feat.shape[1]), dtype=np.float32)
    degree = np.zeros((num_nodes, 1), dtype=np.float32)
    src, dst = edge_index
    np.add.at(features, dst, edge_feat.astype(np.float32))
    np.add.at(degree, dst, 1.0)
    return features / np.maximum(degree, 1.0)


def load_ogbn_node_property(root: str | Path, name: str) -> CitationGraph:
    """Load OGB node property prediction datasets through ogb.nodeproppred."""

    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError("Install `ogb` in the active environment to load OGB datasets.") from exc

    root = Path(root)
    if root.name == "planetoid":
        root = root.parent / "ogb"
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat
    try:
        dataset = NodePropPredDataset(name=name, root=str(root))
    finally:
        torch.load = original_torch_load
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    num_nodes = int(graph["num_nodes"])
    edge_index_np = np.asarray(graph["edge_index"], dtype=np.int64)

    node_feat = graph.get("node_feat")
    task_type = "multiclass"
    metric_name = "accuracy"
    if node_feat is None:
        edge_feat = graph.get("edge_feat")
        if edge_feat is None:
            raise ValueError(f"{name} has neither node_feat nor edge_feat.")
        node_feat = _aggregate_edge_features(num_nodes, edge_index_np, np.asarray(edge_feat))

    y_np = np.asarray(labels)
    if name == "ogbn-proteins":
        task_type = "multilabel"
        metric_name = "rocauc"
        y = torch.as_tensor(y_np, dtype=torch.float32)
    else:
        y = torch.as_tensor(y_np.reshape(-1), dtype=torch.long)

    x = torch.as_tensor(np.asarray(node_feat), dtype=torch.float32)
    train_mask = _index_mask(split_idx["train"], num_nodes)
    val_mask = _index_mask(split_idx["valid"], num_nodes)
    test_mask = _index_mask(split_idx["test"], num_nodes)

    return CitationGraph(
        x=x,
        y=y,
        edge_index=torch.as_tensor(edge_index_np, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        task_type=task_type,
        metric_name=metric_name,
    )
