# Assignment 3 — Edge-Aware Weisfeiler–Lehman Kernel

## Research question
Does incorporating chemical bond labels during Weisfeiler–Lehman (WL) neighbourhood refinement improve molecular graph classification on MUTAG over the standard WL subtree kernel?

## What is implemented
- **Baseline:** from-scratch WL subtree kernel with iterations `h = 0, 1, 2, 3, 4`.
- **Modification A:** edge-aware refinement, where each neighbour contribution includes both bond label and neighbour node label.
- **Modification B:** edge-aware refinement with depth-decayed feature counts (`decay=0.7`), giving lower weight to higher-order neighbourhoods.
- **Classifier:** SVM with a precomputed, L2-normalized kernel matrix (`C=1.0`).
- **Evaluation:** repeated stratified 10-fold cross-validation (10 repeats; random seed 42).
- **Metrics:** accuracy, balanced accuracy, macro-F1, ROC-AUC and runtime.

The code implements the WL feature extraction and kernel matrix itself; it does not call a prebuilt WL-kernel package.

## Run
Tested target environment: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/run_experiment.py
```

The script downloads the standard TU Dortmund MUTAG archive to `data/MUTAG.zip`, then writes:
- `results/metrics.csv`
- `results/summary.json`
- `results/results_report.md`

If the download is blocked, download `MUTAG.zip` from the graph-kernel dataset page linked below and place it in `data/` before running.

## How to interpret the comparison
Compare `WL subtree (h=3)` against `Edge-aware WL (h=3)`. They use the same graphs, cross-validation splits, classifier, and metrics. This isolates the effect of including bond labels in neighbourhood signatures. The `decay=0.7` experiment tests a second modification: reducing the contribution of deeper WL iterations.

Do not assume the modification must improve every metric. Report the measured outcome, including negative or negligible changes. MUTAG has only 188 graphs, so describe conclusions as evidence on this benchmark rather than a universal claim.

## Research references
1. Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). “Weisfeiler-Lehman Graph Kernels.” *Journal of Machine Learning Research*, 12, 2539–2561. https://jmlr.org/papers/v12/shervashidze11a.html
2. Morris, C., Kriege, N. M., Kersting, K., & Mutzel, P. (2016). “Faster Kernels for Graphs with Continuous Attributes via Hashing.” *IEEE ICDM 2016*, 1095–1100. https://doi.org/10.1109/ICDM.2016.0142
3. Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B., & Borgwardt, K. (2019). “Wasserstein Weisfeiler-Lehman Graph Kernels.” *NeurIPS 2019*. https://proceedings.neurips.cc/paper/2019/hash/73fed7fd472e502d8908794430511f4d-Abstract.html
4. Debnath, A. K., et al. (1991). “Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.” *Journal of Medicinal Chemistry*, 34(2), 786–797. https://doi.org/10.1021/jm00106a046

## Important scientific note
Edge-aware refinement is a research-motivated, straightforward extension to discrete edge labels; do not claim it is a newly invented kernel. The original WL paper notes that extension to discrete edge labels is straightforward. The contribution here is the controlled implementation and empirical ablation on MUTAG.

## Generate the hardcopy report
After the experiment completes successfully, run:

```bash
python src/make_pdf_report.py
```

This writes `report/WL_Kernel_Assignment_Report.pdf` using the actual `results/metrics.csv` values. Print that PDF for the hardcopy submission. Do not submit a report until the dataset experiment has run and the table contains measured results.
