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

# STEP 2 (optional): auto-download accessible literature files (best effort)
!python colab_single_cell_runner.py --output_dir "/content" --download_dir "/content/downloaded_literature" --download_only

# STEP 3: after auto-download (or manual download), upload/organize supplementary/article tables in /content/uploaded_literature_tables
# Optional metadata CSV (/content/uploaded_metadata.csv) columns:
# filename,reference,citation,source_url

# STEP 4: extract uploaded files into canonical dataset + run modeling/publication evaluation
!python colab_single_cell_runner.py --output_dir "/content" --uploaded_dir "/content/uploaded_literature_tables" --uploaded_metadata_csv "/content/uploaded_metadata.csv"

# Optional: save outputs to Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v /content/reference_candidates.csv /content/graphene_polymer_literature_matrix.csv /content/graphene_polymer_literature_matrix.xlsx parity_plot.png shap_summary.png feature_importance_shap_bar.png /content/drive/MyDrive/
```

Notes:
- `reference_candidates.csv` includes `DOI` and `Download_URL` columns.
- Auto-download is best-effort and only retrieves publicly reachable files; paywalled content may remain undownloaded in `download_report.csv`.
- Uploaded-file extraction currently supports `.csv`, `.xls`, `.xlsx`, `.html`, `.htm` tables.
