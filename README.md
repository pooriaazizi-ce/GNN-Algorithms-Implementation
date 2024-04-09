# Implementation of Basic Graph Neural Networks Algorithms

This repository contains implementations of basic graph neural networks Algorithms, including Graph Attention Networks v2 (GATv2), Graph Attention Networks (GAT), and Graph Convolutional Networks (GCN), along with their application on the CoraFull and CiteSeer datasets using the PyTorch Geometric library.

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy

## Overview of Notebook Content

### Part a)
- Downloading CoraFull and CiteSeer datasets.
- Reporting dataset statistics including number of classes, nodes, edges, and number of features per node.
- Dividing the dataset into training, validation, and test sets.

### Part b)
- Implementing an MLP(Multi Layer Perceptron) model with 4 layers to classify the feature nodes of each dataset.
- Reporting the accuracy of the architecture.

### Part c)
- Implementing the one-layer, two-layer and three-layer GCN model and finding the best number of hidden layer dimensions based on validation set.
- Reporting the best GCN model's performance on the test set.

- Implementing the one-layer, two-layer and three-layer GAT model and finding the best number of hidden layer dimensions based on validation set.
- Reporting the best GAT model's performance on the test set.

### Part e)
- Implementing the one-layer, two-layer and three-layer GATv2 model and finding the best number of hidden layer dimensions based on validation set.
- Reporting the best GATv2 model's performance on the test set.


## Related Papers

- Graph Convolutional Networks (GCN): [Semi-Supervised Classification with Graph Convolutional Networks 2016](https://arxiv.org/abs/1609.02907)
- Graph Attention Networks (GAT): [Graph Attention Networks 2017](https://arxiv.org/abs/1710.10903)
- Graph Attention Networks v2 (GATv2): [How Attentive are Graph Attention Networks? 2021](https://arxiv.org/abs/2105.14491)


## Contributions

Contributions are welcome. Feel free to submit a pull request for any improvements or additional functionality.
