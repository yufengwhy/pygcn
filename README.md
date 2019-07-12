## Pygcn w/o pifall
**This pygcn implementation is the same as TensorFlow implementation** in https://github.com/tkipf/gcn, fixing the subtle differences of [data splits, normalization, dropout](https://github.com/tkipf/pygcn/issues/20) in author's pygcn https://github.com/tkipf/pygcn.

##  Performance
- cora: 0.820 (paper: 0.815)
- citeseer: 0.707 (paper: 0.703)
- pubmed: 0.794 (paper: 0.790)

## Usage

```python train.py --dataset cora --early_stopping 10```  
```python train.py --dataset citeseer --early_stopping 10```  
```python train.py --dataset pubmed --early_stopping 20```
**early_stopping can be 10 for cora and citeseer, 20 for pubmed.**

## References
PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].
[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6
