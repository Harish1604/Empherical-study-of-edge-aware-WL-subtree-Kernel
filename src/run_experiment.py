#!/usr/bin/env python3
"""From-scratch WL subtree-kernel experiment on MUTAG.

Downloads the standard TU Dortmund MUTAG archive, implements standard WL
subtree features and an edge-aware WL variant, evaluates precomputed-kernel
SVMs with repeated stratified 10-fold CV, and writes CSV/JSON/Markdown outputs.
"""
from __future__ import annotations
import io, json, time, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
URLS = [
    "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip",
    "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip",
]


def download_mutag() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "MUTAG.zip"
    if target.exists() and target.stat().st_size > 1000:
        return target
    last_error = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 WL-kernel-assignment"})
            with urllib.request.urlopen(req, timeout=25) as response:
                blob = response.read()
            if not blob.startswith(b"PK"):
                raise ValueError("The server did not return a ZIP archive")
            target.write_bytes(blob)
            return target
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Could not download MUTAG automatically. Download MUTAG.zip from the TU graph-kernel "
        "dataset page and place it in the project's data/ folder. Last error: " + repr(last_error)
    )


def read_lines(zf: zipfile.ZipFile, suffix: str) -> list[str]:
    names = zf.namelist()
    match = next((n for n in names if n.replace('\\', '/').endswith(suffix)), None)
    if match is None:
        raise FileNotFoundError(f"{suffix} not found inside MUTAG.zip; archive files: {names[:10]}")
    return zf.read(match).decode("utf-8").strip().splitlines()


def load_mutag(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        edges = [tuple(map(int, x.split(','))) for x in read_lines(zf, "MUTAG_A.txt")]
        graph_indicator = [int(x) for x in read_lines(zf, "MUTAG_graph_indicator.txt")]
        y = [int(x) for x in read_lines(zf, "MUTAG_graph_labels.txt")]
        node_labels = [int(x) for x in read_lines(zf, "MUTAG_node_labels.txt")]
        edge_labels = [int(x) for x in read_lines(zf, "MUTAG_edge_labels.txt")]
    n_graphs = len(y)
    graphs = [nx.Graph() for _ in range(n_graphs)]
    node_to_graph = {}
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        g = graph_id - 1
        node_to_graph[node_id] = g
        graphs[g].add_node(node_id - 1, label=str(node_labels[node_id - 1]))
    if len(edges) != len(edge_labels):
        raise ValueError(f"Edge count ({len(edges)}) does not match edge labels ({len(edge_labels)})")
    for (u, v), edge_label in zip(edges, edge_labels):
        gu, gv = node_to_graph[u], node_to_graph[v]
        if gu != gv:
            raise ValueError("An edge crosses graph boundaries in MUTAG data")
        graphs[gu].add_edge(u - 1, v - 1, label=str(edge_label))
    # TU graph labels are typically -1/+1. Convert to 0/1 for metric consistency.
    labels = np.array([1 if v > 0 else 0 for v in y], dtype=int)
    return graphs, labels


def _compress_signatures(signatures_by_graph):
    """Map equal signatures to equal integer IDs consistently across all graphs."""
    unique = sorted({sig for per_graph in signatures_by_graph for sig in per_graph.values()}, key=repr)
    mapping = {sig: idx for idx, sig in enumerate(unique)}
    return [{node: mapping[sig] for node, sig in per_graph.items()} for per_graph in signatures_by_graph]


def wl_feature_dicts(graphs, height=3, edge_aware=False, decay=1.0):
    """Return per-graph sparse WL subtree count features (no external WL library).

    Standard variant: each new label hashes (own label, sorted neighbour labels).
    Edge-aware variant: each new label hashes (own label, sorted (bond label, neighbour label)).
    The same label dictionary is shared across graphs at each iteration.
    """
    labels = [{v: str(data.get("label", g.degree(v))) for v, data in g.nodes(data=True)}
              for g in graphs]
    feature_dicts = [defaultdict(float) for _ in graphs]
    for h in range(height + 1):
        for gi, g in enumerate(graphs):
            weight = float(decay ** h)
            for node, lab in labels[gi].items():
                feature_dicts[gi][(h, lab)] += weight
        if h == height:
            break
        signatures = []
        for gi, g in enumerate(graphs):
            sigs = {}
            for v in g.nodes:
                if edge_aware:
                    neighbourhood = sorted((str(g.edges[v, u].get("label", "UNK")), labels[gi][u]) for u in g.neighbors(v))
                else:
                    neighbourhood = sorted(labels[gi][u] for u in g.neighbors(v))
                sigs[v] = (labels[gi][v], tuple(neighbourhood))
            signatures.append(sigs)
        labels = _compress_signatures(signatures)
    return [dict(d) for d in feature_dicts]


def kernel_from_features(feature_dicts):
    vocab = sorted({key for fd in feature_dicts for key in fd}, key=repr)
    col = {key: i for i, key in enumerate(vocab)}
    X = np.zeros((len(feature_dicts), len(vocab)), dtype=float)
    for i, fd in enumerate(feature_dicts):
        for key, value in fd.items():
            X[i, col[key]] = value
    # L2-normalized feature maps keep graph size from dominating similarity.
    X = normalize(X, norm="l2")
    return X @ X.T


def evaluate_kernel(K, y, cv):
    """Average metrics over all repeated CV test folds; no test fold is reused as training data."""
    fold_metrics = []
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        model = SVC(kernel="precomputed", C=1.0)
        model.fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
        pred = model.predict(K[np.ix_(test_idx, train_idx)])
        scores = model.decision_function(K[np.ix_(test_idx, train_idx)])
        row = {
            "accuracy": accuracy_score(y[test_idx], pred),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, average="macro", zero_division=0),
        }
        row["roc_auc"] = roc_auc_score(y[test_idx], scores) if len(np.unique(y[test_idx])) == 2 else np.nan
        fold_metrics.append(row)
    frame = pd.DataFrame(fold_metrics)
    out = {}
    for col in frame.columns:
        out[col] = float(frame[col].mean())
        out[col + "_std"] = float(frame[col].std(ddof=1))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = download_mutag()
    graphs, y = load_mutag(zip_path)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    experiments = []
    configs = []
    for h in (0, 1, 2, 3, 4):
        configs.append((f"WL subtree (h={h})", h, False, 1.0))

    configs.append(("Edge-aware WL (h=3)", 3, True, 1.0))
    
    configs.append(("Edge-aware WL (h=3, decay=0.7)", 3, True, 0.7))
    for name, h, edge_aware, decay in configs:
        t0 = time.perf_counter()
        features = wl_feature_dicts(graphs, height=h, edge_aware=edge_aware, decay=decay)
        K = kernel_from_features(features)
        # CV object is reproducibly seeded; instantiate fresh for each experiment.
        cv_this = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
        metrics = evaluate_kernel(K, y, cv_this)
        elapsed = time.perf_counter() - t0
        experiments.append({"method": name, "height": h, "edge_aware": edge_aware,
                            "decay": decay, "runtime_seconds": elapsed, **metrics})
        print(f"{name:34s} acc={metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f} macro-F1={metrics['macro_f1']:.4f}±{metrics['macro_f1_std']:.4f} AUC={metrics['roc_auc']:.4f}")
    df = pd.DataFrame(experiments)
    df.to_csv(OUT_DIR / "metrics.csv", index=False)
    summary = {
        "dataset": "MUTAG", "n_graphs": len(graphs), "n_nodes": int(sum(g.number_of_nodes() for g in graphs)),
        "n_edges": int(sum(g.number_of_edges() for g in graphs)),
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "cv": "RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)",
        "classifier": "SVC(kernel='precomputed', C=1.0)",
        "results": experiments,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    best = df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    base = df[df.method == "WL subtree (h=3)"].iloc[0]
    modified = df[df.method == "Edge-aware WL (h=3)"].iloc[0]
    report = f"""# Experimental Results (generated by run_experiment.py)\n\nDataset: MUTAG ({len(graphs)} graphs; {sum(g.number_of_nodes() for g in graphs)} nodes; {sum(g.number_of_edges() for g in graphs)} edges).\n\nProtocol: repeated stratified 10-fold cross-validation (10 repeats, seed 42); precomputed-kernel SVM with C=1.0. WL labels and feature vectors are constructed without using class labels.\n\n## Results\n\n{df.to_markdown(index=False, floatfmt='.4f')}\n\n## Main comparison\n\n- Standard WL subtree (h=3): accuracy {base.accuracy:.4f} ± {base.accuracy_std:.4f}, macro-F1 {base.macro_f1:.4f} ± {base.macro_f1_std:.4f}, ROC-AUC {base.roc_auc:.4f} ± {base.roc_auc_std:.4f}.\n- Edge-aware WL (h=3): accuracy {modified.accuracy:.4f} ± {modified.accuracy_std:.4f}, macro-F1 {modified.macro_f1:.4f} ± {modified.macro_f1_std:.4f}, ROC-AUC {modified.roc_auc:.4f} ± {modified.roc_auc_std:.4f}.\n- Difference in mean accuracy (edge-aware minus standard): {modified.accuracy-base.accuracy:+.4f}.\n- Difference in mean macro-F1 (edge-aware minus standard): {modified.macro_f1-base.macro_f1:+.4f}.\n\nBest macro-F1 configuration in this run: **{best.method}** (macro-F1 {best.macro_f1:.4f}).\n\nInterpretation: report the measured differences, not an assumed improvement. Because MUTAG is small, results can vary by split; the repeated CV protocol reduces but does not eliminate this uncertainty.\n"""
    (OUT_DIR / "results_report.md").write_text(report)
    print(f"\nWrote: {OUT_DIR/'metrics.csv'}, {OUT_DIR/'summary.json'}, {OUT_DIR/'results_report.md'}")

if __name__ == "__main__":
    main()
