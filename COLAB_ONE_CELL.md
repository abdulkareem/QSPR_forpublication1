```python
# Google Colab (single cell)
!pip install -q pandas numpy scikit-learn xgboost shap mendeleev matplotlib seaborn

# Clone repo
%cd /content
!git clone https://github.com/abdulkareem/QSPR_forpublication1.git || true
%cd /content/QSPR_forpublication1
!git pull

# Run publication workflow (creates dataset + trains model + SHAP plots)
!python colab_publication_workflow.py

# Optional: persist outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -v graphene_pani_pedotpss_dataset_augmented.csv parity_plot_xgb.png shap_summary_xgb.png /content/drive/MyDrive/
```
