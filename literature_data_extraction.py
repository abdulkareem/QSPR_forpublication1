"""Utilities to collect QSPR descriptors from published literature (online).

Designed for Google Colab/free-tier:
- lightweight deps (pandas, requests, beautifulsoup4, lxml)
- table-first extraction from open web pages / supplementary CSV links
- normalization to a common descriptor matrix + reference traceability
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Canonical schema requested for QSPR modeling
CANONICAL_COLUMNS = [
    "Graphene_Surface_Area_m2_g",
    "Polymer_Weight_pct",
    "Doping_Level",
    "Functional_Group_Electronegativity",
    "Polymer_Type",
    "Electrolyte_Type",
    "Specific_Capacitance_F_g",
    "Reference",
    "Source_URL",
]

# Flexible column aliases seen in literature
ALIASES = {
    "Graphene_Surface_Area_m2_g": ["surface area", "bet", "ssa", "m2/g", "m^2/g"],
    "Polymer_Weight_pct": ["polymer wt", "wt%", "weight %", "mass fraction", "loading"],
    "Doping_Level": ["doping", "dopant", "oxidation level", "protonation"],
    "Functional_Group_Electronegativity": ["electronegativity", "x_fg", "functional group en"],
    "Polymer_Type": ["polymer", "conductive polymer", "pani", "ppy", "pedot"],
    "Electrolyte_Type": ["electrolyte", "electrolyte type", "electrolyte solution"],
    "Specific_Capacitance_F_g": ["specific capacitance", "capacitance", "f/g", "f g-1", "f g^-1"],
}


@dataclass
class SourceSpec:
    url: str
    reference: str  # DOI or full citation short-form


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _match_column(colname: str) -> str | None:
    c = _clean_name(colname)
    for target, keys in ALIASES.items():
        if any(k in c for k in keys):
            return target
    return None


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = (
                out[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            )
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def extract_tables_from_url(url: str) -> list[pd.DataFrame]:
    """Download page and extract all HTML tables."""
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    # pandas.read_html is compact and robust for table extraction
    return pd.read_html(io.StringIO(r.text))


def extract_supplement_links(url: str) -> list[str]:
    """Find likely supplementary CSV/XLS/XLSX links on a paper page."""
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.(csv|xls|xlsx)$", href, flags=re.I) or "supp" in href.lower():
            links.append(requests.compat.urljoin(url, href))
    return sorted(set(links))


def normalize_table(df: pd.DataFrame, reference: str, source_url: str) -> pd.DataFrame:
    """Map arbitrary table columns to canonical QSPR descriptor schema."""
    mapped = {}
    for c in df.columns:
        target = _match_column(c)
        if target and target not in mapped:
            mapped[target] = c

    if "Specific_Capacitance_F_g" not in mapped:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    out = pd.DataFrame()
    for tgt, src in mapped.items():
        out[tgt] = df[src]

    # Ensure categorical defaults if absent
    if "Polymer_Type" not in out:
        out["Polymer_Type"] = "Unknown"
    if "Electrolyte_Type" not in out:
        out["Electrolyte_Type"] = "Unknown"

    # Add traceability columns
    out["Reference"] = reference
    out["Source_URL"] = source_url

    num_cols = [
        "Graphene_Surface_Area_m2_g",
        "Polymer_Weight_pct",
        "Doping_Level",
        "Functional_Group_Electronegativity",
        "Specific_Capacitance_F_g",
    ]
    out = _coerce_numeric(out, num_cols)

    # retain rows that at least have target + one major descriptor
    keep = out["Specific_Capacitance_F_g"].notna() & (
        out["Graphene_Surface_Area_m2_g"].notna() | out["Polymer_Weight_pct"].notna()
    )
    out = out.loc[keep].copy()

    # add missing canonical columns
    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[CANONICAL_COLUMNS]


def build_literature_matrix(sources: list[SourceSpec]) -> pd.DataFrame:
    """Create a unified data matrix from multiple online literature sources."""
    collected = []
    for s in sources:
        try:
            tables = extract_tables_from_url(s.url)
        except Exception as exc:
            print(f"Skip {s.url} (download/read_html failed): {exc}")
            continue

        for t in tables:
            norm = normalize_table(t, reference=s.reference, source_url=s.url)
            if len(norm):
                collected.append(norm)

        # Optional: also try direct supplementary tabular links
        try:
            supp_links = extract_supplement_links(s.url)
            for link in supp_links:
                if link.lower().endswith(".csv"):
                    d = pd.read_csv(link)
                elif link.lower().endswith((".xls", ".xlsx")):
                    d = pd.read_excel(link)
                else:
                    continue
                norm = normalize_table(d, reference=s.reference, source_url=link)
                if len(norm):
                    collected.append(norm)
        except Exception:
            pass

    if not collected:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    matrix = pd.concat(collected, ignore_index=True)
    matrix = matrix.drop_duplicates().reset_index(drop=True)
    return matrix


def save_outputs(matrix: pd.DataFrame, base_name: str = "graphene_polymer_literature_matrix"):
    matrix.to_csv(f"{base_name}.csv", index=False)
    matrix.to_excel(f"{base_name}.xlsx", index=False)
    print(f"Saved: {base_name}.csv and {base_name}.xlsx")


if __name__ == "__main__":
    # Replace with open-access article URLs + DOI/citation tags.
    # Tip: start with publisher pages that render tables in HTML or provide supplementary CSV/XLS.
    SOURCES = [
        SourceSpec(url="https://example.com/open-access-paper-1", reference="doi:xx.xxxx/xxxxx1"),
        SourceSpec(url="https://example.com/open-access-paper-2", reference="doi:xx.xxxx/xxxxx2"),
    ]

    matrix = build_literature_matrix(SOURCES)
    print("Rows extracted:", len(matrix))
    print(matrix.head())
    save_outputs(matrix)
