#!/usr/bin/env python3
"""From-scratch Weisfeiler-Lehman (WL) subtree-kernel experiment on MUTAG.

This script:
1. Downloads and loads the MUTAG molecular-graph dataset.
2. Extracts standard WL subtree features and two edge-aware variants.
3. Converts graph features into a normalized Gram/kernel matrix.
4. Evaluates a precomputed-kernel SVM using repeated stratified cross-validation.
5. Saves detailed metrics to CSV, JSON, and Markdown files.

The edge-aware and decay options are experimental variants of feature extraction;
the kernel construction and SVM evaluation pipeline are shared by all variants.
"""

from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC


# Resolve paths relative to the project root, so the script can be run from
# another working directory without changing where input/output files go.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"

# Try the primary MUTAG archive URL first, then use the alternative mirror.
URLS = [
    "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip",
    "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip",
]


def download_mutag() -> Path:
    """Return the local MUTAG ZIP path, downloading it if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "MUTAG.zip"

    # Reuse an existing non-trivial archive instead of downloading it again.
    if target.exists() and target.stat().st_size > 1000:
        return target

    last_error = None
    for url in URLS:
        try:
            # Some dataset hosts reject requests without a User-Agent header.
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 WL-kernel-assignment"},
            )
            with urllib.request.urlopen(request, timeout=25) as response:
                blob = response.read()

            # ZIP files begin with the bytes PK; fail clearly if a webpage/error
            # response was returned instead of the dataset archive.
            if not blob.startswith(b"PK"):
                raise ValueError("The server did not return a ZIP archive")

            target.write_bytes(blob)
            return target
        except Exception as exc:
            # If one mirror fails, try the next one and retain the last error.
            last_error = exc

    raise RuntimeError(
        "Could not download MUTAG automatically. Download MUTAG.zip from the "
        "TU graph-kernel dataset page and place it in the project's data/ folder. "
        f"Last error: {last_error!r}"
    )


def read_lines(zf: zipfile.ZipFile, suffix: str) -> list[str]:
    """Read a named dataset file from the ZIP, matching its filename suffix."""
    names = zf.namelist()
    match = next(
        (name for name in names if name.replace("\\", "/").endswith(suffix)),
        None,
    )
    if match is None:
        raise FileNotFoundError(
            f"{suffix} not found inside MUTAG.zip; archive files: {names[:10]}"
        )

    # Each line represents one item in the corresponding MUTAG data file.
    return zf.read(match).decode("utf-8").strip().splitlines()


def load_mutag(zip_path: Path):
    """Build one NetworkX graph per molecule and return graphs plus class labels."""
    with zipfile.ZipFile(zip_path) as zf:
        # Edge endpoints, graph membership, graph classes, atom labels, bond labels.
        edges = [tuple(map(int, line.split(","))) for line in read_lines(zf, "MUTAG_A.txt")]
        graph_indicator = [int(x) for x in read_lines(zf, "MUTAG_graph_indicator.txt")]
        graph_labels = [int(x) for x in read_lines(zf, "MUTAG_graph_labels.txt")]
        node_labels = [int(x) for x in read_lines(zf, "MUTAG_node_labels.txt")]
        edge_labels = [int(x) for x in read_lines(zf, "MUTAG_edge_labels.txt")]

    # MUTAG has one graph-level target per molecule.
    graphs = [nx.Graph() for _ in range(len(graph_labels))]
    node_to_graph = {}

    # The source data uses 1-based node IDs and graph IDs. NetworkX node IDs
    # here are changed to 0-based IDs, while graph IDs are converted to indices.
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graph_index = graph_id - 1
        node_to_graph[node_id] = graph_index
        graphs[graph_index].add_node(
            node_id - 1,
            label=str(node_labels[node_id - 1]),  # Initial atom/node label.
        )

    # Each edge must have exactly one corresponding bond label.
    if len(edges) != len(edge_labels):
        raise ValueError(
            f"Edge count ({len(edges)}) does not match edge labels ({len(edge_labels)})"
        )

    for (u, v), edge_label in zip(edges, edge_labels):
        graph_u = node_to_graph[u]
        graph_v = node_to_graph[v]

        # A valid molecular bond must connect nodes within the same molecule.
        if graph_u != graph_v:
            raise ValueError("An edge crosses graph boundaries in MUTAG data")

        # Store the bond type as an edge attribute. Edge-aware WL uses this label.
        graphs[graph_u].add_edge(u - 1, v - 1, label=str(edge_label))

    # MUTAG graph labels are commonly -1/+1. Convert them to 0/1 for metrics.
    labels = np.array([1 if value > 0 else 0 for value in graph_labels], dtype=int)
    return graphs, labels


def _compress_signatures(signatures_by_graph):
    """Assign the same integer ID to identical node-neighbourhood signatures.

    The mapping is shared across all graphs for this WL iteration. Therefore,
    identical signatures in different graphs receive the same compressed label.
    """
    unique_signatures = sorted(
        {signature for per_graph in signatures_by_graph for signature in per_graph.values()},
        key=repr,
    )
    signature_to_id = {
        signature: index for index, signature in enumerate(unique_signatures)
    }
    return [
        {node: signature_to_id[signature] for node, signature in per_graph.items()}
        for per_graph in signatures_by_graph
    ]


def wl_feature_dicts(graphs, height=3, edge_aware=False, decay=1.0):
    """Extract WL subtree-count features for every graph.

    Parameters
    ----------
    graphs : list[networkx.Graph]
        Input graphs with node labels and, optionally, edge/bond labels.
    height : int
        Maximum WL refinement iteration. height=0 uses only initial node labels.
    edge_aware : bool
        False = standard WL refinement using neighbour node labels only.
        True = edge-aware refinement using (bond label, neighbour node label).
    decay : float
        Weight multiplier per iteration. Feature weight at iteration h is decay**h.
        decay=1.0 means no depth-based down-weighting; decay=0.7 reduces the
        contribution of features from higher WL iterations.

    Returns
    -------
    list[dict]
        One sparse feature-count dictionary per graph. Keys are (iteration, label).
    """
    # Start from the atom labels. The degree is a fallback if a node has no label.
    labels = [
        {node: str(data.get("label", graph.degree(node)))
         for node, data in graph.nodes(data=True)}
        for graph in graphs
    ]

    # Each graph gets a sparse dictionary: (WL iteration, node label) -> count.
    # defaultdict(float) allows fractional counts when decay is less than 1.
    feature_dicts = [defaultdict(float) for _ in graphs]

    # h=0 records the original labels; each following iteration refines them
    # using the labels in the node's neighbourhood from the previous iteration.
    for h in range(height + 1):
        for graph_index, graph in enumerate(graphs):
            # Depth-decay weighting: 1.0**h leaves all depths equally weighted;
            # 0.7**h progressively down-weights deeper WL features.
            weight = float(decay ** h)

            # Count how often each node label occurs at this refinement depth.
            for node, label in labels[graph_index].items():
                feature_dicts[graph_index][(h, label)] += weight

        # Stop after recording features at the requested final height.
        if h == height:
            break

        signatures = []
        for graph_index, graph in enumerate(graphs):
            node_signatures = {}
            for node in graph.nodes:
                if edge_aware:
                    # EDGE-AWARE MODIFICATION:
                    # Include both the bond label and the neighbour's current
                    # WL label. This distinguishes neighbourhoods with different
                    # bond types even if their neighbouring node labels match.
                    neighbourhood = sorted(
                        (
                            str(graph.edges[node, neighbour].get("label", "UNK")),
                            labels[graph_index][neighbour],
                        )
                        for neighbour in graph.neighbors(node)
                    )
                else:
                    # ORIGINAL WL VERSION:
                    # Only the current labels of neighbouring nodes are used;
                    # bond/edge labels are not included in the signature.
                    neighbourhood = sorted(
                        labels[graph_index][neighbour]
                        for neighbour in graph.neighbors(node)
                    )

                # New signature = this node's current label + sorted neighbourhood.
                # Sorting makes the signature independent of neighbour ordering.
                node_signatures[node] = (
                    labels[graph_index][node],
                    tuple(neighbourhood),
                )
            signatures.append(node_signatures)

        # Compress signatures into new integer labels for the next iteration.
        labels = _compress_signatures(signatures)

    # Convert defaultdicts to ordinary dictionaries for downstream processing.
    return [dict(feature_dict) for feature_dict in feature_dicts]


def kernel_from_features(feature_dicts):
    """Convert graph feature dictionaries into a normalized kernel matrix."""
    # Build one shared vocabulary of all (iteration, label) features.
    vocabulary = sorted(
        {key for feature_dict in feature_dicts for key in feature_dict},
        key=repr,
    )
    column_index = {key: index for index, key in enumerate(vocabulary)}

    # X[i, j] is the count/weight of feature j in graph i.
    X = np.zeros((len(feature_dicts), len(vocabulary)), dtype=float)
    for graph_index, feature_dict in enumerate(feature_dicts):
        for feature, value in feature_dict.items():
            X[graph_index, column_index[feature]] = value

    # L2-normalize each graph's feature vector so larger graphs do not dominate
    # similarity solely because they contain more nodes/features.
    X = normalize(X, norm="l2")

    # The Gram matrix K[i,j] is the dot-product similarity between graphs i and j.
    # This is the precomputed kernel matrix passed to the SVM.
    return X @ X.T


def evaluate_kernel(K, y, cv):
    """Evaluate a precomputed graph kernel using the supplied CV splitter."""
    fold_metrics = []

    for train_indices, test_indices in cv.split(np.zeros(len(y)), y):
        # The SVM expects similarities between training examples for fitting.
        model = SVC(kernel="precomputed", C=1.0)
        model.fit(K[np.ix_(train_indices, train_indices)], y[train_indices])

        # For prediction, each test graph needs similarities to training graphs.
        test_train_kernel = K[np.ix_(test_indices, train_indices)]
        predictions = model.predict(test_train_kernel)
        decision_scores = model.decision_function(test_train_kernel)

        # Record classification metrics for this fold.
        row = {
            "accuracy": accuracy_score(y[test_indices], predictions),
            "balanced_accuracy": balanced_accuracy_score(y[test_indices], predictions),
            "macro_f1": f1_score(
                y[test_indices], predictions, average="macro", zero_division=0
            ),
        }

        # ROC-AUC requires both classes to be present in the test fold.
        row["roc_auc"] = (
            roc_auc_score(y[test_indices], decision_scores)
            if len(np.unique(y[test_indices])) == 2
            else np.nan
        )
        fold_metrics.append(row)

    # Report the mean and sample standard deviation across all CV test folds.
    frame = pd.DataFrame(fold_metrics)
    summary = {}
    for column in frame.columns:
        summary[column] = float(frame[column].mean())
        summary[column + "_std"] = float(frame[column].std(ddof=1))
    return summary


def main():
    """Run each WL configuration, evaluate it, and save all result files."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download and parse the molecular graphs and their binary class labels.
    zip_path = download_mutag()
    graphs, y = load_mutag(zip_path)

    # Use 10 folds repeated 10 times. The fixed seed makes the splits reproducible.
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    experiments = []
    configs = []

    # ORIGINAL WL BASELINES: test refinement heights 0 through 4.
    # False = do not use bond labels; 1.0 = no depth decay.
    for h in (0, 1, 2, 3, 4):
        configs.append((f"WL subtree (h={h})", h, False, 1.0))

    # MODIFICATION 1: edge-aware WL at height 3, with no depth decay.
    # Parameters are (display name, height, edge_aware, decay).
    configs.append(("Edge-aware WL (h=3)", 3, True, 1.0))

    # MODIFICATION 2: edge-aware WL at height 3, with depth decay 0.7**h.
    configs.append(("Edge-aware WL (h=3, decay=0.7)", 3, True, 0.7))

    # Run feature extraction, kernel construction, and CV evaluation for each setup.
    for name, h, edge_aware, decay in configs:
        start_time = time.perf_counter()

        # The parameters determine whether edge labels and depth decay are used.
        features = wl_feature_dicts(
            graphs, height=h, edge_aware=edge_aware, decay=decay
        )
        K = kernel_from_features(features)

        # Instantiate a fresh splitter with the same seed for a fair comparison.
        cv_this = RepeatedStratifiedKFold(
            n_splits=10, n_repeats=10, random_state=42
        )
        metrics = evaluate_kernel(K, y, cv_this)
        elapsed = time.perf_counter() - start_time

        # Save configuration details and all averaged metrics for this experiment.
        experiments.append(
            {
                "method": name,
                "height": h,
                "edge_aware": edge_aware,
                "decay": decay,
                "runtime_seconds": elapsed,
                **metrics,
            }
        )

        # Print the main metrics in a compact form for the terminal.
        print(
            f"{name:34s} "
            f"acc={metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f} "
            f"macro-F1={metrics['macro_f1']:.4f}±{metrics['macro_f1_std']:.4f} "
            f"AUC={metrics['roc_auc']:.4f}"
        )

    # Convert experiment records into a table and save as CSV for analysis.
    df = pd.DataFrame(experiments)
    df.to_csv(OUT_DIR / "metrics.csv", index=False)

    # Save dataset metadata, CV protocol, classifier, and results as JSON.
    summary = {
        "dataset": "MUTAG",
        "n_graphs": len(graphs),
        "n_nodes": int(sum(graph.number_of_nodes() for graph in graphs)),
        "n_edges": int(sum(graph.number_of_edges() for graph in graphs)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
        "cv": "RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)",
        "classifier": "SVC(kernel='precomputed', C=1.0)",
        "results": experiments,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Select the best row by macro-F1 (accuracy breaks ties), then compare the
    # standard height-3 baseline with the edge-aware height-3 variant.
    best = df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    base = df[df.method == "WL subtree (h=3)"].iloc[0]
    modified = df[df.method == "Edge-aware WL (h=3)"].iloc[0]

    # Create a readable Markdown report using the measured results.
    report = f"""# Experimental Results (generated by run_experiment.py)

Dataset: MUTAG ({len(graphs)} graphs; {sum(g.number_of_nodes() for g in graphs)} nodes; {sum(g.number_of_edges() for g in graphs)} edges).

Protocol: repeated stratified 10-fold cross-validation (10 repeats, seed 42); precomputed-kernel SVM with C=1.0. WL labels and feature vectors are constructed without using class labels.

## Results

{df.to_markdown(index=False, floatfmt='.4f')}

## Main comparison

- Standard WL subtree (h=3): accuracy {base.accuracy:.4f} ± {base.accuracy_std:.4f}, macro-F1 {base.macro_f1:.4f} ± {base.macro_f1_std:.4f}, ROC-AUC {base.roc_auc:.4f} ± {base.roc_auc_std:.4f}.
- Edge-aware WL (h=3): accuracy {modified.accuracy:.4f} ± {modified.accuracy_std:.4f}, macro-F1 {modified.macro_f1:.4f} ± {modified.macro_f1_std:.4f}, ROC-AUC {modified.roc_auc:.4f} ± {modified.roc_auc_std:.4f}.
- Difference in mean accuracy (edge-aware minus standard): {modified.accuracy-base.accuracy:+.4f}.
- Difference in mean macro-F1 (edge-aware minus standard): {modified.macro_f1-base.macro_f1:+.4f}.

Best macro-F1 configuration in this run: **{best.method}** (macro-F1 {best.macro_f1:.4f}).

Interpretation: report the measured differences, not an assumed improvement. Because MUTAG is small, results can vary by split; repeated cross-validation reduces but does not eliminate this uncertainty.
"""
    (OUT_DIR / "results_report.md").write_text(report)

    print(
        f"\nWrote: {OUT_DIR / 'metrics.csv'}, "
        f"{OUT_DIR / 'summary.json'}, {OUT_DIR / 'results_report.md'}"
    )


if __name__ == "__main__":
    # Run the experiment only when this file is executed directly, not imported.
    main()
