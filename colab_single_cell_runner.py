"""Single-cell style runner for Google Colab.

Usage in Colab (one cell):
!python colab_single_cell_runner.py
"""

import subprocess
import sys
from pathlib import Path


def ensure_packages():
    pkgs = ["numpy", "pandas", "matplotlib", "seaborn", "scikit-learn", "xgboost", "shap", "requests", "beautifulsoup4", "lxml", "openpyxl"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def main():
    ensure_packages()

    from literature_data_extraction import SourceSpec, build_literature_matrix, save_outputs
    from colab_qspr_workflow import run_workflow

    # Replace with real OPEN-ACCESS paper landing pages or supplementary-data pages.
    sources = [
        SourceSpec(url="https://example.com/open-access-paper-1", reference="doi:xx.xxxx/xxxxx1"),
        SourceSpec(url="https://example.com/open-access-paper-2", reference="doi:xx.xxxx/xxxxx2"),
    ]

    matrix = build_literature_matrix(sources)
    if matrix.empty:
        raise RuntimeError(
            "No rows extracted. Update `sources` with real open-access URLs containing HTML tables or supplementary CSV/XLS files."
        )

    save_outputs(matrix, base_name="graphene_polymer_literature_matrix")

    csv_path = Path("graphene_polymer_literature_matrix.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Expected graphene_polymer_literature_matrix.csv not found")

    df, metrics, _ = run_workflow(local_csv=str(csv_path))
    print("\nExtraction + modeling complete.")
    print("Rows used:", len(df))
    print(metrics)


if __name__ == "__main__":
    main()
