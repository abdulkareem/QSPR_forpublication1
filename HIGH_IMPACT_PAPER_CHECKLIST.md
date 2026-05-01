# High-Impact Paper Checklist (JMR Focus Issue)

## Must-have scientific elements
- Clear novelty statement: literature-extracted + citation-traceable QSPR + explainable AI.
- External/temporal validation (not only random split).
- Uncertainty quantification (bootstrap CI for R2/RMSE).
- Mechanistic interpretation linking SHAP/PDP to ion/electron transport physics.

## Required figures (beyond parity + SHAP)
1. Descriptor distributions.
2. Correlation heatmap.
3. Electrolyte-dependent capacitance distributions.
4. Polymer loading response curve with optimum window.
5. Permutation importance chart.
6. Partial dependence plots for top descriptors.
7. Temporal validation performance chart.
8. Ablation study chart/table.

## Reproducibility package
- Data file with `Reference`, `Citation`, `Source_URL` per row.
- `table_references.csv` mapped to manuscript bibliography.
- Exact software versions and random seeds.
- Colab notebook + scripts as supplementary material.

## Sustainability framing
- State reduced experimental waste via ML pre-screening.
- Discuss earth-abundant conductive polymers vs heavy-metal oxide reliance.
- Include estimate of experiments avoided using model-guided selection.

## Use this script for advanced results
Run:
```bash
python high_impact_results.py --data graphene_polymer_literature_matrix.csv --outdir high_impact_outputs
```
