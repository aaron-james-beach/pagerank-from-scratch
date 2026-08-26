"""Power-iteration PageRank over an edge list, without a graph library.

The edge file is streamed once per iteration rather than loaded into an
adjacency matrix. Memory stays proportional to the number of nodes, not the
number of edges, which is what makes a multi-million-edge graph tractable on
a laptop. The cost is re-reading the file each pass — see benchmark.py for
what that trade actually buys.

Handles dangling nodes (nodes with no outgoing edges) explicitly. Without
that, rank mass leaks out of the graph every iteration and the resulting
vector is not a probability distribution.
"""

import time


class PageRank:
    def __init__(self, edge_file, verbose=True):
        self.edge_file = edge_file
        self.verbose = verbose

        self.node_degree = {}
        self.max_node_id = 0
        self.dangling_nodes = None

    def read_edge_file(self):
        """Yield (source, target) pairs one at a time."""
        with open(self.edge_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                source, target = line.split()[:2]
                yield int(source), int(target)

    def calculate_node_degree(self):
        """Count out-degree per node and find the largest node id.

        One pass over the file. Nodes that never appear as a source are
        dangling — they receive rank but have nowhere to send it.
        """
        for source, target in self.read_edge_file():
            if source not in self.node_degree:
                self.node_degree[source] = 1
            else:
                self.node_degree[source] += 1

            if source > self.max_node_id:
                self.max_node_id = source
            if target > self.max_node_id:
                self.max_node_id = target

        n = self.max_node_id + 1
        self.dangling_nodes = [i for i in range(n) if i not in self.node_degree]

        if self.verbose:
            print("Max node id: {}".format(self.max_node_id))
            print("Nodes with outgoing edges: {}".format(len(self.node_degree)))
            print("Dangling nodes: {}".format(len(self.dangling_nodes)))

    def get_max_node_id(self):
        return self.max_node_id

    def run_pagerank(self, node_weights, damping_factor=0.85,
                     iterations=100, tolerance=1e-8, handle_dangling=True):
        """Run power iteration until the rank vector stops moving.

        node_weights is the teleport distribution. Uniform gives standard
        PageRank; any other distribution gives the personalized variant.

        handle_dangling=False reproduces the naive version, which leaks rank
        mass through dead-end nodes. Kept so the difference can be measured
        rather than asserted — see benchmark.py.

        Returns (pr_values, history) where history holds the L1 change at
        each iteration.
        """
        n = self.max_node_id + 1
        pr_values = [1.0 / n] * n
        history = []
        start_time = time.time()

        for it in range(iterations):
            new_pr_values = [0.0] * n

            # Push each node's rank along its outgoing edges, split evenly.
            for source, target in self.read_edge_file():
                new_pr_values[target] += pr_values[source] / self.node_degree[source]

            # Rank sitting on dangling nodes has nowhere to go. Redistribute
            # it across the teleport distribution instead of losing it.
            dangling_mass = 0.0
            if handle_dangling:
                for node in self.dangling_nodes:
                    dangling_mass += pr_values[node]

            for node in range(n):
                teleport = node_weights[node]
                new_pr_values[node] = (
                    (1 - damping_factor) * teleport
                    + damping_factor * (new_pr_values[node] + dangling_mass * teleport)
                )

            delta = 0.0
            for node in range(n):
                delta += abs(new_pr_values[node] - pr_values[node])
            history.append(delta)

            pr_values = new_pr_values

            if delta < tolerance:
                break

        if self.verbose:
            print("Converged after {0} iterations ({1:.1f}s), "
                  "final L1 delta {2:.2e}, total rank mass {3:.6f}".format(
                      it + 1, time.time() - start_time, history[-1], sum(pr_values)))

        return pr_values, history


def uniform_weights(n):
    """Teleport uniformly — standard PageRank."""
    return [1.0 / n] * n


def seeded_weights(n, seed_nodes):
    """Teleport only to a chosen set — personalized PageRank.

    The random walk restarts inside the seed set instead of anywhere in the
    graph, so rank concentrates in the neighborhood around those nodes.
    """
    weights = [0.0] * n
    if not seed_nodes:
        raise ValueError("seed_nodes must not be empty")
    share = 1.0 / len(seed_nodes)
    for node in seed_nodes:
        weights[node] = share
    return weights


def top_n(pr_values, n=10):
    """Return the n highest-ranked (node_id, value) pairs."""
    return sorted(enumerate(pr_values), key=lambda x: x[1], reverse=True)[:n]
