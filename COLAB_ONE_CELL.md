```python
# Google Colab (single cell)
!pip install -q pandas numpy scikit-learn xgboost shap mendeleev matplotlib seaborn

%cd /content
import os
if not os.path.isdir('/content/QSPR_forpublication1'):
    !git clone https://github.com/abdulkareem/QSPR_forpublication1.git
%cd /content/QSPR_forpublication1
!git pull

# STEP 1: list required literature candidates with DOI/download links (no modeling)
!python colab_single_cell_runner.py --output_dir "/content" --list_only
# -> creates /content/reference_candidates.csv

# STEP 2: after manual download, upload supplementary/article tables to /content/uploaded_literature_tables
# Optional metadata CSV (/content/uploaded_metadata.csv) columns:
# filename,reference,citation,source_url

# STEP 3: extract uploaded files into canonical dataset + run modeling/publication evaluation
!python colab_single_cell_runner.py --output_dir "/content" --uploaded_dir "/content/uploaded_literature_tables" --uploaded_metadata_csv "/content/uploaded_metadata.csv"

# Optional: save outputs to Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v /content/reference_candidates.csv /content/graphene_polymer_literature_matrix.csv /content/graphene_polymer_literature_matrix.xlsx parity_plot.png shap_summary.png feature_importance_shap_bar.png /content/drive/MyDrive/
```

Notes:
- `reference_candidates.csv` includes `DOI` and `Download_URL` columns for manual downloading.
- Uploaded-file extraction currently supports `.csv`, `.xls`, `.xlsx`, `.html`, `.htm` tables.
