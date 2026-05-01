"""Generate extended analysis outputs for manuscript-quality figures.

Run after dataset creation:
python paper_analysis_and_figures.py --data graphene_polymer_literature_matrix.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV matrix produced by extraction/workflow")
    p.add_argument("--outdir", default="paper_figures", help="Output directory for figures/tables")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    # basic cleaning
    num_cols = [
        "Graphene_Surface_Area_m2_g",
        "Polymer_Weight_pct",
        "Doping_Level",
        "Functional_Group_Electronegativity",
        "Specific_Capacitance_F_g",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Figure 1: descriptor distributions
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=160)
    cols_for_hist = [c for c in num_cols if c in df.columns][:6]
    for ax, c in zip(axes.ravel(), cols_for_hist):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#2a9d8f")
        ax.set_title(c)
    for ax in axes.ravel()[len(cols_for_hist):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_descriptor_distributions.png", bbox_inches="tight")

    # Figure 2: correlation heatmap
    corr = df[[c for c in num_cols if c in df.columns]].corr(numeric_only=True)
    plt.figure(figsize=(7.5, 6), dpi=160)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Descriptor Correlation Matrix")
    plt.tight_layout()
    plt.savefig(outdir / "fig2_correlation_heatmap.png", bbox_inches="tight")

    # Figure 3: capacitance by electrolyte
    if "Electrolyte_Type" in df.columns:
        plt.figure(figsize=(8, 5.8), dpi=160)
        sns.boxplot(data=df, x="Electrolyte_Type", y="Specific_Capacitance_F_g", palette="Set2")
        sns.stripplot(data=df, x="Electrolyte_Type", y="Specific_Capacitance_F_g", color="k", alpha=0.35, size=3)
        plt.title("Specific Capacitance by Electrolyte")
        plt.tight_layout()
        plt.savefig(outdir / "fig3_capacitance_by_electrolyte.png", bbox_inches="tight")

    # Figure 4: capacitance vs polymer loading
    if "Polymer_Weight_pct" in df.columns:
        plt.figure(figsize=(7.8, 6), dpi=160)
        sns.scatterplot(data=df, x="Polymer_Weight_pct", y="Specific_Capacitance_F_g", hue="Polymer_Type", alpha=0.8)
        sns.regplot(data=df, x="Polymer_Weight_pct", y="Specific_Capacitance_F_g", scatter=False, order=2, color="red")
        plt.title("Capacitance vs Polymer Loading")
        plt.tight_layout()
        plt.savefig(outdir / "fig4_capacitance_vs_loading.png", bbox_inches="tight")

    # Figure 5: surface area vs capacitance, sized by doping
    if {"Graphene_Surface_Area_m2_g", "Doping_Level"}.issubset(df.columns):
        plt.figure(figsize=(8, 6), dpi=160)
        sns.scatterplot(
            data=df,
            x="Graphene_Surface_Area_m2_g",
            y="Specific_Capacitance_F_g",
            hue="Polymer_Type" if "Polymer_Type" in df.columns else None,
            size="Doping_Level",
            sizes=(20, 200),
            alpha=0.7,
        )
        plt.title("Capacitance vs Surface Area (bubble size: doping)")
        plt.tight_layout()
        plt.savefig(outdir / "fig5_surface_area_bubble.png", bbox_inches="tight")

    # Figure 6: publication trend if year exists
    if "Publication_Year" in df.columns:
        yearly = df.groupby("Publication_Year", dropna=True).size()
        plt.figure(figsize=(8.2, 4.8), dpi=160)
        yearly.plot(kind="bar", color="#457b9d")
        plt.ylabel("Number of records")
        plt.title("Literature Record Distribution by Year")
        plt.tight_layout()
        plt.savefig(outdir / "fig6_yearly_records.png", bbox_inches="tight")

    # Table summary
    summary = df.describe(include="all").T
    summary.to_csv(outdir / "table_descriptor_summary.csv")

    # Reference list export for manuscript SI
    ref_cols = [c for c in ["Reference", "Citation", "Source_URL"] if c in df.columns]
    if ref_cols:
        refs = df[ref_cols].drop_duplicates().sort_values(ref_cols[0])
        refs.to_csv(outdir / "table_references.csv", index=False)

    print(f"Saved figures and tables in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
