```python
# ===== Single Colab Cell: extract literature data online + run QSPR analysis =====
!pip -q install numpy pandas matplotlib seaborn scikit-learn xgboost shap requests beautifulsoup4 lxml openpyxl

# 1) Ensure the two scripts are available in your Colab runtime.
#    Option A: upload `literature_data_extraction.py` and `colab_qspr_workflow.py` manually.
#    Option B: clone/download your repo first.

from literature_data_extraction import SourceSpec, build_literature_matrix, save_outputs
from colab_qspr_workflow import run_workflow

# 2) Provide real open-access article pages or supplementary-data links.
sources = [
    SourceSpec(url="https://example.com/open-access-paper-1", reference="doi:xx.xxxx/xxxxx1"),
    SourceSpec(url="https://example.com/open-access-paper-2", reference="doi:xx.xxxx/xxxxx2"),
]

# 3) Extract, normalize, and save the literature matrix.
matrix = build_literature_matrix(sources)
if matrix.empty:
    raise RuntimeError("No rows extracted. Replace sources with valid open-access URLs that include tables/supplementary files.")

save_outputs(matrix, base_name="graphene_polymer_literature_matrix")

# 4) Run model workflow using extracted CSV.
df, metrics_df, best_model = run_workflow(local_csv="graphene_polymer_literature_matrix.csv")

print("\nDone. Rows in matrix:", len(df))
print(metrics_df)
```
