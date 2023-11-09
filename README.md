# Repo Packer

## What Does It Do?
RepoPacker optimizes the order of files in training data to provide the best training signal for a Language Model. It groups semantically connected files—such as definitions, implementations, and usages—into closely-clustered sequences and arranges them appropriately. For instance, in Python repositories,
files like `main.py`, `module/__init__.py`, and `module/somestuff.py` are reordered to place `module/somestuff.py` followed by `main.py` if `main.py` references `module/somestuff.py` through
an import statement.

Ordering the files such that the definition file precedes the referencing file is crucial. This sequence closely mirrors the scenario during code completion, potentially providing significant benefits to the training process.

### Additional Features:
* Language-agnostic and universally applicable
* Capable of aligning documentation and configuration files with the corresponding source code
* Efficiently reorders large repositories containing thousands of files

#### Check out `Usage example.ipynb` for a practical demonstration!

## The Algorithm

The algorithm treats files in a repository as nodes and organizes the process to find an optimal Hamiltonian path through a complete graph. The graph is initially populated with edge weights derived from the BM25 score between the contents of different files.

The optimization algorithm, implemented in the `TSPSolver` class, aims to solve a variation of the Traveling Salesperson Problem (TSP) by finding a Hamiltonian path that maximizes the total weight of the edges, often referred to as the "Maximum Weight Hamiltonian Path Problem" (MWHPP). The algorithm proceeds as follows:

1. **Initial Clustering:**
   - Applies the Hungarian algorithm to condense the complete graph into a sparse graph with simple chains and cycles.

2. **Cycle Handling:**
   - Identifies and breaks cycles within the graph by removing the least-weight edges.

3. **Chain Merging:**
   - Iteratively merges the best pairs of chains, prioritizing those with the highest connecting edge weight, to form a single continuous chain.

4. **Result:**
   - Outputs the final chain representing the maximum weight Hamiltonian path that visits each node exactly once.

This algorithm offers a heuristic solution to the NP-hard problem of finding a maximum weight Hamiltonian path by adopting a greedy approach to merge chains based on edge weights, ensuring an order that leads to an optimal solution.