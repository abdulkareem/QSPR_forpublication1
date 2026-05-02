"""Single-command runner for Google Colab.

Example:
!python colab_single_cell_runner.py --output_dir "/content"
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def ensure_packages() -> None:
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
    p.add_argument("--download_dir", default=None, help="Optional folder to auto-download accessible papers/supplements")
    p.add_argument("--download_only", action="store_true", help="Only download candidate literature files and exit")
    p.add_argument("--dataset_csv", default=None, help="Direct curated dataset CSV (skip discovery/extraction and run analysis)")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown arguments: {unknown}")
    return args


def _find_csv_candidates() -> list[Path]:
    roots = [Path.cwd(), Path("/content"), Path("/content/drive/MyDrive")]
    found: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        found.extend(sorted(r.glob("*.csv")))
    # unique preserve order
    uniq = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _resolve_dataset_csv(path_arg: str) -> Path:
    p = Path(path_arg).expanduser()
    if p.exists():
        return p.resolve()

    # try common Colab roots by basename
    name = p.name
    for root in [Path.cwd(), Path("/content"), Path("/content/drive/MyDrive")]:
        cand = root / name
        if cand.exists():
            return cand.resolve()

    csvs = _find_csv_candidates()
    hint = "\n".join([f" - {c}" for c in csvs[:20]]) if csvs else "(no CSV files found in /content or /content/drive/MyDrive)"
    raise FileNotFoundError(
        f"dataset_csv not found: {path_arg}\n"
        f"Tip: upload file then pass exact path, e.g. --dataset_csv /content/<your_file>.csv\n"
        f"Detected CSV files:\n{hint}"
    )


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
        download_sources,
        save_outputs,
    )
    from colab_qspr_workflow import run_workflow
    from colab_publication_workflow import run_publication_workflow

    if args.dataset_csv:
        csv_path = _resolve_dataset_csv(args.dataset_csv)
        print(f"Using direct curated dataset: {csv_path}")
        df, metrics, _ = run_workflow(local_csv=str(csv_path))
        pub_report = run_publication_workflow(local_csv=str(csv_path))
        print("\nAnalysis complete from direct dataset CSV.")
        print("Rows used:", len(df))
        print(metrics)
        print("Publication report:", pub_report)
        return

    if args.download_dir:
        if args.references_csv:
            dl_sources = load_sources_from_csv(args.references_csv)
        else:
            dl_sources = search_crossref_sources(year_from=1996, year_to=2026, rows=80)
            save_reference_candidates(dl_sources, csv_path=str(outdir / "reference_candidates.csv"))
        report = download_sources(dl_sources, outdir=args.download_dir)
        print(report["Download_Status"].value_counts(dropna=False))
        print(f"Download report saved at: {Path(args.download_dir) / 'download_report.csv'}")
        if args.download_only:
            return

    if args.uploaded_dir:
        matrix = build_literature_matrix_from_uploaded_files(args.uploaded_dir, metadata_csv=args.uploaded_metadata_csv)
        print(f"Loaded uploaded files from: {args.uploaded_dir}")
    else:
        if args.references_csv:
            sources = load_sources_from_csv(args.references_csv)
            print(f"Loaded curated references: {len(sources)} from {args.references_csv}")
        else:
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
    print("Publication report:", pub_report)
    print(f"\nOutputs saved under: {outdir}")
if __name__ == "__main__":
    main()

