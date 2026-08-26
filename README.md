# PageRank From Scratch

Power-iteration PageRank over a five-million-edge web graph, implemented
without a graph library. The edge list is streamed from disk once per
iteration instead of being loaded into an adjacency matrix, so memory scales
with the number of nodes rather than the number of edges.

Includes the personalized variant, a correctness check against NetworkX, and
a measurement of what happens when dangling nodes are ignored.

Run against the [SNAP web-Google graph](https://snap.stanford.edu/data/web-Google.html):
875,713 nodes, 5,105,039 edges.

## The dangling node problem

Many nodes in a crawled web graph have no outgoing edges — pages that were
linked to but never crawled, or that link nowhere.

On this graph, 136,259 of 875,713 nodes (15.6%) have no outgoing edges.

A naive implementation pushes rank along outgoing edges and adds the teleport
term. Dangling nodes have no outgoing edges, so the rank sitting on them
disappears each iteration:

| | Total rank mass at convergence |
|---|---|
| Ignoring dangling nodes | **0.673** |
| Redistributing dangling mass | **1.000** |

A vector summing to 0.673 is not a probability distribution, and the values
cannot be read as "probability a random surfer lands here."

The interesting part is what it *doesn't* break:

| | Top-1000 |
|---|---|
| Ranking agreement between the two | 1000/1000 |

The leak behaves close to a uniform rescaling, so if you only care about
*which* nodes rank highest, the naive version gives the same answer. If you
care about the values — comparing across graphs, thresholding, feeding them
to a downstream model — it does not.

Both paths are implemented. `handle_dangling=False` reproduces the naive
behavior so the difference can be measured rather than asserted. The implementation indexes by node id, so a graph with sparse, high-numbered
ids allocates a rank vector larger than the node count. web-Google's ids run
to 916,427 across 875,713 actual nodes, so roughly 40,700 slots hold no node
and behave as dangling — harmless here, since they receive only teleport mass
and redistribute it, but it inflates the raw dangling count the loader
reports. Remapping ids to a contiguous range would fix it at the cost of a
lookup table.

## Damping factor and convergence

Convergence is on the L1 norm of the change in the rank vector, tolerance
1e-8, rather than a fixed iteration count. Iterations required grow sharply
with the damping factor:

| Damping | Iterations to converge |
|---|---|
| 0.50 | 22 |
| 0.65 | 35 |
| 0.75 | 51 |
| 0.85 | 89 |
| 0.90 | 137 |
| 0.95 | did not converge within 200 |

This is expected — the error contracts by roughly the damping factor each
iteration, so the iteration count scales like 1/log(1/damping) and blows up
as damping approaches 1. The practical consequence is that a fixed iteration
budget is not a safe substitute for a convergence check: 10 or 25 iterations
at damping 0.85 stops well short of the fixed point on this graph.

## Correctness

Checked against `networkx.pagerank` on the same graph: top-100 rankings agree
exactly. Residual value differences come from a node-indexing choice — this
implementation indexes by node id up to the maximum observed, while NetworkX
indexes only nodes appearing in an edge, so the uniform teleport vector is
spread over slightly different denominators.

## Personalized PageRank

Standard PageRank teleports uniformly. The personalized variant restricts the
teleport distribution to a chosen seed set, so the random walk restarts
inside that neighborhood and rank concentrates around it. Same power
iteration, different teleport vector.

```python
from pagerank import PageRank, seeded_weights, top_n

pr = PageRank("data/web-Google.txt")
pr.calculate_node_degree()

weights = seeded_weights(pr.get_max_node_id() + 1, seed_nodes=[41909, 504140])
ranks, history = pr.run_pagerank(weights)
print(top_n(ranks, 10))
```

## Data

Any whitespace-separated edge list of integer node ids will work; lines
beginning with `#` are skipped.

```bash
cd data
curl -O https://snap.stanford.edu/data/web-Google.txt.gz
gunzip web-Google.txt.gz
cd ..
```

## Run

```bash
pip install -r requirements.txt

python src/benchmark.py data/web-Google.txt --skip-networkx
```

Drop `--skip-networkx` to include the correctness check. NetworkX builds the
whole graph in memory, which is the constraint this implementation was
written to avoid, so expect it to need considerably more RAM than the
streaming version.

## Design note

Streaming the edge file on every iteration is a deliberate trade. It keeps
memory at O(nodes) instead of O(edges), which is what lets a five-million-edge
graph run on a laptop with no sparse matrix backend. The cost is disk I/O
proportional to the iteration count.

For a graph that fits comfortably in memory, building a CSR sparse matrix
once and doing matrix-vector products would be substantially faster. The
streaming version wins when the edge list does not fit, and it keeps the
algorithm legible: each iteration is visibly "push each node's rank along its
edges, then teleport."

## Notes

Rank is split evenly across a node's outgoing edges. Weighted edges would
require dividing by the sum of weights rather than the out-degree.

Duplicate edges are counted, not deduplicated — an edge appearing twice
carries twice the rank. On a crawl-derived graph that is usually intended;
on other data it may not be.

The implementation indexes by node id, so a graph with sparse, high-numbered
ids allocates a rank vector larger than the node count. Remapping to a
contiguous range would fix that at the cost of a lookup table.
