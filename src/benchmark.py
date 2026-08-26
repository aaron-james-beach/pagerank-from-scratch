"""Correctness and behavior checks for the from-scratch implementation.

Three things the assignment never asked for:

1. Agreement with networkx.pagerank, to show the implementation is right
   rather than merely plausible.
2. A damping-factor sweep, to show how the convergence rate depends on it.
3. A measurement of what dangling-node handling actually changes — which
   turns out to be the values, not the rankings.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagerank import PageRank, uniform_weights, top_n  # noqa: E402


def compare_with_networkx(edge_file, damping=0.85, top_k=100):
    """Check our rankings against networkx on the same graph."""
    import networkx as nx

    graph = nx.DiGraph()
    with open(edge_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            source, target = line.split()[:2]
            graph.add_edge(int(source), int(target))

    reference = nx.pagerank(graph, alpha=damping, tol=1e-10)

    pr = PageRank(edge_file)
    pr.calculate_node_degree()
    weights = uniform_weights(pr.get_max_node_id() + 1)
    ours, _ = pr.run_pagerank(weights, damping_factor=damping, tolerance=1e-10)

    ours_top = [node for node, _ in top_n(ours, top_k)]
    ref_top = [node for node, _ in sorted(
        reference.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    overlap = len(set(ours_top) & set(ref_top))
    max_diff = max(abs(ours[node] - reference.get(node, 0.0))
                   for node in ref_top)

    print("Top-{0} overlap with networkx: {1}/{0}".format(top_k, overlap))
    print("Largest value difference in that set: {0:.2e}".format(max_diff))
    return {"top_k": top_k, "overlap": overlap, "max_value_diff": max_diff}


def damping_sweep(edge_file, dampings=(0.50, 0.65, 0.75, 0.85, 0.90, 0.95)):
    """How many iterations to converge, as a function of damping."""
    pr = PageRank(edge_file, verbose=False)
    pr.calculate_node_degree()
    weights = uniform_weights(pr.get_max_node_id() + 1)

    rows = []
    for damping in dampings:
        _, history = pr.run_pagerank(weights, damping_factor=damping,
                                     iterations=200, tolerance=1e-8)
        rows.append({"damping": damping, "iterations": len(history)})
        print("damping {0:.2f} -> {1} iterations".format(damping, len(history)))
    return rows


def dangling_effect(edge_file, damping=0.85, top_k=1000):
    """What does redistributing dangling mass actually change?"""
    pr = PageRank(edge_file, verbose=False)
    pr.calculate_node_degree()
    weights = uniform_weights(pr.get_max_node_id() + 1)

    fixed, _ = pr.run_pagerank(weights, damping_factor=damping)
    naive, _ = pr.run_pagerank(weights, damping_factor=damping,
                               handle_dangling=False)

    fixed_top = {node for node, _ in top_n(fixed, top_k)}
    naive_top = {node for node, _ in top_n(naive, top_k)}

    result = {
        "rank_mass_with_handling": sum(fixed),
        "rank_mass_without": sum(naive),
        "top_k": top_k,
        "top_k_overlap": len(fixed_top & naive_top),
    }
    print("Rank mass with dangling handling:    {0:.6f}".format(
        result["rank_mass_with_handling"]))
    print("Rank mass without:                   {0:.6f}".format(
        result["rank_mass_without"]))
    print("Top-{0} overlap between the two:  {1}/{0}".format(
        top_k, result["top_k_overlap"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file")
    parser.add_argument("--skip-networkx", action="store_true",
                        help="networkx needs the whole graph in memory")
    parser.add_argument("--out", default="figures/benchmark_results.json")
    args = parser.parse_args()

    results = {}

    print("=== Dangling node handling ===")
    results["dangling"] = dangling_effect(args.edge_file)

    print("\n=== Damping factor sweep ===")
    results["damping_sweep"] = damping_sweep(args.edge_file)

    if not args.skip_networkx:
        print("\n=== Agreement with networkx ===")
        results["networkx"] = compare_with_networkx(args.edge_file)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote {}".format(args.out))


if __name__ == "__main__":
    main()
