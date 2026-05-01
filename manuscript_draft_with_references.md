# Draft Manuscript (JMR Focus Issue)

## Title
**Nanoscale Pathways in Graphene–Conductive Polymer Nanocomposites: Literature-Extracted QSPR Modeling of Specific Capacitance with Explainable AI**

## Abstract
We present a literature-extracted and citation-traceable machine learning workflow for predicting specific capacitance (F/g) of graphene-conductive polymer nanocomposites. Records are aggregated from online sources with per-row provenance (`Reference`, `Citation`, `Source_URL`), standardized, and modeled using tree-based learning with explainability. The resulting model performance (report final test $R^2$ and RMSE from your run) and SHAP interpretation identify polymer loading, electrolyte class, and accessible surface area as major design drivers. The framework supports greener materials development by reducing wasteful trial-and-error synthesis.

## 1. Introduction
- Supercapacitor electrodes based on graphene/polymer composites remain promising due to high power capability and tunable interfacial kinetics.
- Conductive polymers (e.g., PANI, PEDOT:PSS) provide pseudocapacitive contributions while avoiding dependence on scarce heavy-metal oxide chemistries.
- A data-centric route is needed to accelerate sustainable optimization.

## 2. Methods
1. Online literature discovery and extraction over the target period.
2. Canonical descriptor mapping and unit harmonization.
3. QSPR modeling with cross-validated hyperparameter tuning.
4. SHAP-based interpretation.

## 3. Results (figures to include)
1. **Descriptor distributions**: `fig1_descriptor_distributions.png`
2. **Correlation matrix**: `fig2_correlation_heatmap.png`
3. **Capacitance vs electrolyte**: `fig3_capacitance_by_electrolyte.png`
4. **Capacitance vs polymer loading**: `fig4_capacitance_vs_loading.png`
5. **Surface area–capacitance bubble plot**: `fig5_surface_area_bubble.png`
6. **Yearly record distribution**: `fig6_yearly_records.png`
7. **Model parity + SHAP** from training workflow (`parity_plot_xgb.png`, `shap_summary_xgb.png`)

### Interpretation guidance
- Discuss optimal intermediate polymer wt% where pore accessibility and redox-site density are balanced.
- Show how electrolyte class shifts utilization efficiency.
- Use SHAP ranking to propose candidate synthesis windows.

## 4. Discussion
- Compare trends with established graphene/PANI and graphene/PEDOT:PSS reports.
- Quantify how data-driven screening can cut experimental iterations.
- Address limitations: extraction bias toward open-access/machine-readable sources.

## 5. Conclusion
- Citation-traceable literature matrix + explainable QSPR offers a publishable framework for identifying nanoscale pathways.
- Workflow is Colab-ready and reproducible with exportable references table.

## 6. References and Supporting Information
- Use `paper_figures/table_references.csv` for manuscript bibliography mapping.
- Include data dictionary and model settings in SI.
