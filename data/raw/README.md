# Raw Data

`planetoid/` stores the raw Planetoid citation-network files downloaded from:

https://github.com/kimiyoung/planetoid

The loader reads these raw files directly and preserves the standard Planetoid split.

Amazon Photo is downloaded on demand from the public gnn-benchmark NPZ URL.

OGBN datasets are downloaded on demand through `ogb.nodeproppred.NodePropPredDataset`.
Install the `ogb` Python package and keep network access enabled for the first run, or
pre-populate the OGB raw files under `data/raw/ogb/`.
