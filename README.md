# Geopolitical Shocks and Green Bond Pricing

## Cross-Country Evidence from News and Markets

Research materials accompanying the manuscript. Current revision: 17 November 2026.

## Manuscript

- [Main article (Word)](manuscript/Geopolitical_Shocks_Green_Bond_Pricing.docx)
- [Supplementary analysis (Word)](manuscript/Geopolitical_Shocks_Green_Bond_Supplement.docx)
- [Main article text](manuscript/manuscript.md) (document-build markers retained; formatted equations, tables and figures are in the Word article).

The current article contains five editable, centred equations with right-aligned numbers, four figures, five main tables and 25 references. The supplement contains 16 tables.

## Research design and findings

The study combines sampled geopolitical news, an international issuer-matched bond panel, contractual verification, local yield curves and yield-implied price reconciliation around the 28 February 2026 escalation involving Iran. The market window is 14 January–3 April 2026.

The broad panel contains 74 pairs, seven currencies and 4,207 pair-date observations. Its relative quoted-price coefficient is approximately −0.092 log percentage points. Identical-sample curve conditioning reduces the EUR/USD estimate. Detailed Australian and New Zealand sovereign cases display different conditional patterns. A joint model directly estimates their difference; its direction is stable across the reported grid, while precision depends on calendar and covariance choices.

These are event-associated price comparisons and conditional case diagnostics, not a generally identified causal green-label penalty. Price basis points are not yield basis points. See the article and supplement for the complete estimates and inference assumptions.

## Repository contents

- `manuscript/`: latest article and supplementary document.
- `green2026/revision/`: estimation scripts and writing/measurement documentation.
- `green2026/revision/outputs/`: selected aggregate estimates and validation results.
- `green2026/revision/figures/`: all four article figures.
- `green2026/revision/inputs/literature_2024_2026/`: verified citation metadata and source register.

## Reproduce the comparative figures without licensed data

Run from the repository root with Python 3:

```bash
python -m pip install -r requirements.txt
python green2026/revision/make_comparative_figures.py
```

This regenerates Figures 3 and 4 from the included aggregate model estimates. Figure 4 displays all 24 direct contrasts, including imprecise specifications. Figures 1 and 2 are supplied as manuscript outputs.

## Full estimation and data access

Instrument-level LSEG data, quote histories, private analytical panels and the private replication ZIP are intentionally excluded. This repository is not a self-contained reproduction of every regression. The original licensed data owner retains the complete checkpoint package. A valid data licence and the original compatible input files are needed for full re-estimation.

The archived analysis scripts preserve their original relative paths. They are intended to run in the complete licensed workspace in this sequence:

```bash
python green2026/revision/analyze_revision.py
python green2026/revision/enhance_analysis.py
python green2026/revision/joint_inference.py
python green2026/revision/coupon_audit.py
python green2026/revision/contract_audit5.py
python green2026/revision/core_rates_analysis.py
python green2026/revision/cross_case_test.py
```

Public contractual and central-bank source links appear in the supplement and `sources.json`. Files ending in `_PRIVATE` are omitted; public-source files needed by the original scripts must also be supplied at their documented paths. Do not replace missing observations with simulated data and present the resulting estimates as this study's results.

## Availability

This is a private research repository. The link does not grant access to readers or reviewers who have not been authorised. No open-source or redistribution licence is granted by this deposit. Third-party data remain governed by their respective terms.
