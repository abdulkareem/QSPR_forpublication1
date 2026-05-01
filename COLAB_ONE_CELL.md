```python
# Google Colab (single cell)
!pip install -q pandas numpy scikit-learn xgboost shap mendeleev matplotlib seaborn

# Clone repo only if missing; otherwise pull latest
%cd /content
import os
if not os.path.isdir('/content/QSPR_forpublication1'):
    !git clone https://github.com/abdulkareem/QSPR_forpublication1.git
%cd /content/QSPR_forpublication1
!git pull

# 1) First pass: auto-discover candidate references and save /content/reference_candidates.csv
!python colab_single_cell_runner.py --output_dir "/content"

# 2) If extraction failed, edit /content/reference_candidates.csv manually (replace landing pages with direct article/supplement URLs),
#    then upload as /content/my_references.csv and rerun with:
# !python colab_single_cell_runner.py --output_dir "/content" --references_csv "/content/my_references.csv"

# Optional: persist outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v /content/reference_candidates.csv /content/graphene_polymer_literature_matrix.csv parity_plot.png shap_summary.png feature_importance_shap_bar.png /content/drive/MyDrive/
```

Note: reference CSV columns expected are `URL`, `Reference`, and optional `Citation`.
