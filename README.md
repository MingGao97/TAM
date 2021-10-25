# TAM: Learning DAG with Testing and Masking

This is a `python` implementation of the following paper:

[1] Gao, M., Aragam, B. (2021). [Efficient Bayesian network structure learning via local Markov boundary search](https://arxiv.org/abs/2110.06082) ([NeurIPS 2021](https://nips.cc/Conferences/2021/)).

If you find this code useful, please consider citing:
```
@inproceedings{gao2021tam,
    author = {Gao, Ming and Aragam, Bryon},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{Efficient Bayesian network structure learning via local Markov boundary search}},
    year = {2021}
}
```

## Introduction

The `TAM` algorithm is used for learning the structure of a directed acyclic graph (DAG) `G`. Given data from distribution `P(X)` that satisfies the entropic conditions in the paper above, `TAM` efficiently learns the DAG `G` that generated the samples.

## Requirements
- python
- Package `numpy`
- Package `igraph`
- Package `networkx`

## Contents
- `tam.py` Main function to run the algorithm, see demo below
- `utils.py` Some helper functions to simulate data and evaluate results
- `entropy.py` Mnimiax optimal entropy estimator, see reference
- `coeffs.txt` Best polynomial approximation coefficients used by `entropy.py`

## Demo
Generate a *Tree* graph with 5 nodes. Then generate data by a *MOD* model with *p=0.2* and sample size 1000.
```python
from utils import *
from tam import TAM

G = simulate_dag(d=5, s0=5, graph_type='Tree')
X = sample_from_mod(G=G, n=1000, prob=0.2)
```
Apply algorithm with *kappa=0.005* and *omega=0.001*.
```python
model = TAM(X)
model.train(kappa=0.005, omega=0.001)
```
Print result and SHD with the true graph.
```python
print(model.Gr)
print(count_accuracy(G, model.Gr)['shd'])
```

## References
We adopt the minimax optimal entropy estimator from [here](https://github.com/Albuso0/entropy).
