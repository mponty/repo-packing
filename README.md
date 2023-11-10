# Repo Rearranger


Smarter ordering for optimal Example Packing in the training of autoregressive Language Models


## What Does It Do?

Repo Rearranger is designed to enhance the training of autoregressive
Language Models by optimizing the organization of files in training datasets.
It effectively groups files based on semantic relationships, arranging 
them in sequences that mirror real-world, repository-level code completion
scenarios.

It groups semantically connected files—such
as definitions, implementations, and usages—into closely clustered sequences and arranges them appropriately. For example, in a Python repository with files like `main.py`, `module/__init__.py`, and `module/somestuff.py`, it will reorder 
to place `module/somestuff.py` followed by `main.py` if `main.py` references
`module/somestuff.py` through an import statement.

Such file ordering is important for adapting Language Models to
_**typical code-completion scenario**_, which often involve using 
functions, objects, and classes defined _**earlier**_ in other files of the repository.

### Additional Features:
* **Language-agnostic** and universally applicable.
* Aligns documentation and configuration files with their relevant source code.
* Efficiently reorders large repositories with thousands of files.
* Implements **fuzzy term matching** to recognize semantically similar keywords across different naming conventions, 
 such as **camelCase** or **snake_case**, treating terms like `get_property_name` and `PropertyName` as related, thereby enhancing contextual links within the data.

### Explore [`Usage example.ipynb`](https://github.com/mponty/repo-packing/blob/main/Usage%20example.ipynb) for a practical demonstration!

## The Algorithm

The algorithm treats files in a repository as nodes in a graph,
aiming to find an optimal Hamiltonian path through a complete graph with edge weights derived from `BM25` scores between file contents.

Developed within the `TSPSolver` class, the algorithm addresses a variation of the Traveling Salesperson Problem (TSP) by creating a Hamiltonian path that maximizes the total weight of the edges, known as the Maximum Weight Hamiltonian Path Problem. The process unfolds as follows:

1. **Initial Clustering:**
   - Utilizes the Jonker-Volgenant algorithm to reduce the complete graph into a sparser structure of simple chains and cycles, resulting in one large simple cycle or several connected components, each being either a chain or a simple cycle.

2. **Cycle Handling:**
   - Identifies and eliminates cycles by removing the edges with the least weight, thereby commencing the assembly with multiple simple chains.

3. **Chain Merging:**
   - Repeatedly merges the most advantageous chain pairs, emphasizing those with the highest connecting edge weights, to form a unified chain.

4. **Result:**
   - Produces the final chain, representing a Hamiltonian path with maximum weight, covering each node precisely once.

This algorithm provides a heuristic approach to solve the NP-hard problem of finding a maximum weight Hamiltonian path, using a greedy method for merging chains based on edge weights to achieve an optimal solution.
