```python
# ===== One Colab cell: clone repo, extract literature data, save to Google Drive, run QSPR =====
from google.colab import drive
drive.mount('/content/drive')

!pip -q install numpy pandas matplotlib seaborn scikit-learn xgboost shap requests beautifulsoup4 lxml openpyxl

# 1) Clone your GitHub repo (or pull latest if already cloned)
%cd /content
!git clone https://github.com/abdulkareem/QSPR_forpublication1.git || true
%cd /content/QSPR_forpublication1
!git pull

# 2) OPTIONAL: edit `sources` in colab_single_cell_runner.py OR in literature_data_extraction.py
#    Replace placeholder URLs with real open-access paper/supplement links.

# 3) Run end-to-end pipeline and store outputs in Google Drive
!python colab_single_cell_runner.py --output_dir "/content/drive/MyDrive/QSPR_outputs"

# 4) Check outputs in Drive
!ls -lah "/content/drive/MyDrive/QSPR_outputs"
```
