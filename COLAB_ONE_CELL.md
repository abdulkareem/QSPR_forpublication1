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

# Run automatic online literature search (last 30 years) + extraction + QSPR modeling

!python colab_single_cell_runner.py --output_dir "/content"

# Extraction-first then strict publication evaluation runs inside this command
# Optional: persist outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v /content/graphene_polymer_literature_matrix.csv parity_plot.png shap_summary.png feature_importance_shap_bar.png /content/drive/MyDrive/
```

Note: Extracted matrix includes `Reference`, `Citation`, and `Source_URL` columns for manuscript-ready citation tracking.
