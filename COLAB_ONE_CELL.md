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
# (Now includes graceful fallback if online table extraction returns zero rows)
!python colab_single_cell_runner.py --output_dir "/content"

# Optional: run publication-style augmented workflow too
!python colab_publication_workflow.py

# Optional: persist outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v graphene_pani_pedotpss_dataset_augmented.csv parity_plot_xgb.png shap_summary_xgb.png /content/drive/MyDrive/
```

Note: Extracted matrix includes `Reference`, `Citation`, and `Source_URL` columns for manuscript-ready citation tracking.
