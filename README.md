**This pygcn implementation is the same as TensorFlow implementation** in https://github.com/tkipf/gcn, fixing the subtle differences of [data splits, normalization, dropout](https://github.com/tkipf/pygcn/issues/20) in author's pygcn https://github.com/tkipf/pygcn.

The performance:
- cora: 0.820 (paper: 0.815)
- citeseer: 0.707 (paper: 0.703)
- pubmed: 0.794 (paper: 0.790)

**Note: early_stopping can be 10 for cora and citeseer, 20 for pubmed.**

Graph Convolutional Networks in PyTorch

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)
