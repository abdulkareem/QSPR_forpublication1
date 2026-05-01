"""Single-command runner for Google Colab.

Example (inside repo folder):
!python colab_single_cell_runner.py --output_dir "/content/drive/MyDrive/QSPR_outputs"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_packages():
    pkgs = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "shap",
        "requests",
        "beautifulsoup4",
        "lxml",
        "openpyxl",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=".", help="Folder where extracted matrix and figures are stored")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_packages()

    from literature_data_extraction import SourceSpec, build_literature_matrix, save_outputs
    from colab_qspr_workflow import run_workflow

    sources = [
        SourceSpec(url="https://example.com/open-access-paper-1", reference="doi:xx.xxxx/xxxxx1"),
        SourceSpec(url="https://example.com/open-access-paper-2", reference="doi:xx.xxxx/xxxxx2"),
    ]

    matrix = build_literature_matrix(sources)
    if matrix.empty:
        raise RuntimeError("No rows extracted. Replace `sources` with valid open-access URLs.")

    base = outdir / "graphene_polymer_literature_matrix"
    save_outputs(matrix, base_name=str(base))

    csv_path = outdir / "graphene_polymer_literature_matrix.csv"
    df, metrics, _ = run_workflow(local_csv=str(csv_path))

    print("\nExtraction + modeling complete.")
    print("Rows used:", len(df))
    print(metrics)
    print(f"\nOutputs saved under: {outdir}")


if __name__ == "__main__":
    main()
