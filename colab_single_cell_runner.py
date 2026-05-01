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
    p.add_argument("--references_csv", default=None, help="Optional CSV of curated references with URL/Reference/Citation")
    p.add_argument("--list_only", "--list-only", dest="list_only", action="store_true", help="Only generate reference candidate list, no extraction/modeling")
    p.add_argument("--uploaded_dir", default=None, help="Directory containing uploaded table files (csv/xls/xlsx/html)")
    p.add_argument("--uploaded_metadata_csv", default=None, help="Optional metadata CSV mapping uploaded files to Reference/Citation")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown arguments: {unknown}")
    return args


def main():
    args = parse_args()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_packages()

    from literature_data_extraction import (
        search_crossref_sources,
        load_sources_from_csv,
        save_reference_candidates,
        build_literature_matrix,
        build_literature_matrix_from_uploaded_files,
        save_outputs,
    )
    from colab_qspr_workflow import run_workflow
    from colab_publication_workflow import run_publication_workflow

    if args.uploaded_dir:
        matrix = build_literature_matrix_from_uploaded_files(args.uploaded_dir, metadata_csv=args.uploaded_metadata_csv)
        print(f"Loaded uploaded files from: {args.uploaded_dir}")
    else:
        if args.references_csv:
            sources = load_sources_from_csv(args.references_csv)
            print(f"Loaded curated references: {len(sources)} from {args.references_csv}")
        else:
            # automatic online literature discovery for last 30 years (1996-2026)
            sources = search_crossref_sources(year_from=1996, year_to=2026, rows=80)
            print(f"Discovered candidate sources: {len(sources)}")
            save_reference_candidates(sources, csv_path=str(outdir / "reference_candidates.csv"))
            if args.list_only:
                print(f"List-only mode complete. Edit: {outdir / 'reference_candidates.csv'}")
                return

        matrix = build_literature_matrix(sources)
    if matrix.empty:
        raise RuntimeError("No machine-readable rows extracted. Edit reference_candidates.csv (or provide --references_csv) with publisher/supplement links and rerun.")

    base = outdir / "graphene_polymer_literature_matrix"
    save_outputs(matrix, base_name=str(base))
    csv_path = outdir / "graphene_polymer_literature_matrix.csv"

    df, metrics, _ = run_workflow(local_csv=str(csv_path))
    pub_report = run_publication_workflow(local_csv=str(csv_path))

    print("\nExtraction + modeling complete.")
    print("Rows used:", len(df))
    print(metrics)
    print(f"\nOutputs saved under: {outdir}")


if __name__ == "__main__":
    main()

