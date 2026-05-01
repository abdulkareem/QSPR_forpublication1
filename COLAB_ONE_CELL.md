```python
# Google Colab (single cell)
!pip install -q pandas numpy scikit-learn xgboost shap mendeleev matplotlib seaborn

# Clone repo
%cd /content
!git clone https://github.com/abdulkareem/QSPR_forpublication1.git || true
%cd /content/QSPR_forpublication1
!git pull

# Run automatic online literature search (last 30 years) + extraction + QSPR modeling
!python colab_single_cell_runner.py --output_dir "/content"

# (Optional) run publication-style augmented workflow too
!python colab_publication_workflow.py

# Optional: persist outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v graphene_pani_pedotpss_dataset_augmented.csv parity_plot_xgb.png shap_summary_xgb.png /content/drive/MyDrive/
```
