# AdaptiveHAC
Modified Hierarchical Agglomerative Clustering that can adaptively learn the number of cluster

## How to use
This mean that there is no need to pass the parameter: `n_cluster`

A new parameter is added: `threhold`

`threhold` determinded the max distance between any two cluster, that is, the distance between any returned two cluster in no less than threhold.

##Example
Just run:

```
python test_cluster_example.py
```
