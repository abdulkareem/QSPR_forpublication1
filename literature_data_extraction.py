"""Online literature discovery + tabular extraction for QSPR dataset building.

This module now supports:
1) Searching published literature online (Crossref API) for the last 30 years.
2) Identifying candidate article URLs/DOIs.
3) Extracting HTML/supplement tables and normalizing into canonical matrix columns.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

CANONICAL_COLUMNS = [
    "Graphene_Surface_Area_m2_g",
    "Polymer_Weight_pct",
    "Doping_Level",
    "Functional_Group_Electronegativity",
    "Polymer_Type",
    "Electrolyte_Type",
    "Specific_Capacitance_F_g",
    "Reference",
    "Citation",
    "Source_URL",
]

ALIASES = {
    "Graphene_Surface_Area_m2_g": ["surface area", "bet", "ssa", "m2/g", "m^2/g"],
    "Polymer_Weight_pct": ["polymer wt", "wt%", "weight %", "mass fraction", "loading"],
    "Doping_Level": ["doping", "dopant", "oxidation level", "protonation"],
    "Functional_Group_Electronegativity": ["electronegativity", "x_fg", "functional group en"],
    "Polymer_Type": ["polymer", "conductive polymer", "pani", "pedot"],
    "Electrolyte_Type": ["electrolyte", "electrolyte type", "electrolyte solution"],
    "Specific_Capacitance_F_g": ["specific capacitance", "capacitance", "f/g", "f g-1", "f g^-1"],
}


@dataclass
class SourceSpec:
    url: str
    reference: str
    citation: str = ""


def search_crossref_sources(
    query: str = "graphene conductive polymer supercapacitor specific capacitance PANI PEDOT",
    year_from: int = 1996,
    year_to: int = 2026,
    rows: int = 60,
) -> list[SourceSpec]:
    """Discover article URLs/DOIs from Crossref for the last 30 years."""
    endpoint = "https://api.crossref.org/works"
    params = {
        "query": query,
        "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31",
        "rows": rows,
        "select": "DOI,URL,title,container-title,author,issued",
    }
    r = requests.get(endpoint, params=params, timeout=45)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])

    out = []
    for it in items:
        doi = it.get("DOI")
        url = it.get("URL")
        if not url:
            continue
        ref = f"doi:{doi}" if doi else url
        citation = _format_crossref_citation(it)
        out.append(SourceSpec(url=url, reference=ref, citation=citation))

    # unique by URL
    uniq = {}
    for s in out:
        uniq[s.url] = s
    return list(uniq.values())


def load_sources_from_csv(path: str) -> list[SourceSpec]:
    """Load user-curated references from CSV with columns: URL, Reference, Citation."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "url" not in cols:
        raise ValueError("references CSV must include a URL column")

    out = []
    for _, row in df.iterrows():
        url = str(row[cols["url"]]).strip()
        if not url or url.lower() == "nan":
            continue
        ref_col = cols.get("reference")
        cit_col = cols.get("citation")
        ref = str(row[ref_col]).strip() if ref_col else url
        if not ref or ref.lower() == "nan":
            ref = url
        citation = str(row[cit_col]).strip() if cit_col else ""
        if citation.lower() == "nan":
            citation = ""
        out.append(SourceSpec(url=url, reference=ref, citation=citation))

    # unique by URL preserving first appearance
    uniq = {}
    for src in out:
        if src.url not in uniq:
            uniq[src.url] = src
    return list(uniq.values())


def save_reference_candidates(sources: list[SourceSpec], csv_path: str = "reference_candidates.csv") -> None:
    rows = []
    for s in sources:
        doi = s.reference.replace("doi:", "") if str(s.reference).startswith("doi:") else ""
        rows.append({
            "Reference": s.reference,
            "URL": s.url,
            "DOI": doi,
            "Download_URL": f"https://doi.org/{doi}" if doi else s.url,
            "Citation": s.citation,
            "Use_for_manual_download": "YES",
            "Notes": "Replace URL with direct article/supplement table links if needed",
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved reference candidate list: {csv_path}")


def build_literature_matrix_from_uploaded_files(upload_dir: str, metadata_csv: str | None = None) -> pd.DataFrame:
    """Extract tabular data from uploaded CSV/XLS/XLSX/HTML files and attach citations."""
    base = Path(upload_dir)
    files = sorted([p for p in base.glob("**/*") if p.suffix.lower() in {".csv", ".xls", ".xlsx", ".html", ".htm"}])
    if not files:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    meta = {}
    if metadata_csv:
        mdf = pd.read_csv(metadata_csv)
        lower = {c.lower(): c for c in mdf.columns}
        name_col = lower.get("filename") or lower.get("file")
        if name_col:
            for _, r in mdf.iterrows():
                key = str(r[name_col]).strip()
                meta[key] = {
                    "reference": str(r[lower.get("reference")]).strip() if lower.get("reference") else key,
                    "citation": str(r[lower.get("citation")]).strip() if lower.get("citation") else "",
                    "source_url": str(r[lower.get("source_url")]).strip() if lower.get("source_url") else key,
                }

    collected = []
    for fp in files:
        entry = meta.get(fp.name, {"reference": fp.name, "citation": "", "source_url": fp.name})
        try:
            if fp.suffix.lower() == ".csv":
                tables = [pd.read_csv(fp)]
            elif fp.suffix.lower() in {".xls", ".xlsx"}:
                tables = [pd.read_excel(fp)]
            else:
                tables = pd.read_html(str(fp))
        except Exception:
            continue

        for t in tables:
            norm = normalize_table(t, entry["reference"], entry["source_url"], entry["citation"])
            if len(norm):
                collected.append(norm)

    if not collected:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(collected, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _format_crossref_citation(item: dict) -> str:
    title = (item.get("title") or [""])[0]
    container = (item.get("container-title") or [""])[0]
    year = ""
    issued = item.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        year = str(issued[0][0])

    authors = item.get("author", [])
    a_txt = ""
    if authors:
        names = []
        for a in authors[:3]:
            fam = a.get("family", "")
            giv = a.get("given", "")
            names.append((fam + (", " + giv if giv else "")).strip(", "))
        a_txt = "; ".join(names)
        if len(authors) > 3:
            a_txt += " et al."

    doi = item.get("DOI", "")
    parts = [p for p in [a_txt, year, title, container, f"doi:{doi}" if doi else ""] if p]
    return ". ".join(parts)

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
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def extract_supplement_links(url: str) -> list[str]:
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.(csv|xls|xlsx)$", href, flags=re.I) or "supp" in href.lower():
            links.append(requests.compat.urljoin(url, href))
    return sorted(set(links))


def normalize_table(df: pd.DataFrame, reference: str, source_url: str, citation: str = "") -> pd.DataFrame:
    mapped = {}
    for c in df.columns:
        t = _match_column(c)
        if t and t not in mapped:
            mapped[t] = c

    if "Specific_Capacitance_F_g" not in mapped:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    out = pd.DataFrame()
    for tgt, src in mapped.items():
        out[tgt] = df[src]

    if "Polymer_Type" not in out:
        out["Polymer_Type"] = "Unknown"
    if "Electrolyte_Type" not in out:
        out["Electrolyte_Type"] = "Unknown"

    out["Reference"] = reference
    out["Citation"] = citation
    out["Source_URL"] = source_url

    num_cols = [
        "Graphene_Surface_Area_m2_g",
        "Polymer_Weight_pct",
        "Doping_Level",
        "Functional_Group_Electronegativity",
        "Specific_Capacitance_F_g",
    ]
    out = _coerce_numeric(out, num_cols)

    keep = out["Specific_Capacitance_F_g"].notna() & (
        out["Graphene_Surface_Area_m2_g"].notna() | out["Polymer_Weight_pct"].notna()
    )
    out = out.loc[keep].copy()

    for c in CANONICAL_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[CANONICAL_COLUMNS]


def build_literature_matrix(sources: list[SourceSpec]) -> pd.DataFrame:
    collected = []
    for s in sources:
        try:
            tables = extract_tables_from_url(s.url)
        except Exception:
            continue

        for t in tables:
            norm = normalize_table(t, s.reference, s.url, s.citation)
            if len(norm):
                collected.append(norm)

        try:
            supp = extract_supplement_links(s.url)
            for link in supp:
                if link.lower().endswith(".csv"):
                    d = pd.read_csv(link)
                elif link.lower().endswith((".xls", ".xlsx")):
                    d = pd.read_excel(link)
                else:
                    continue
                norm = normalize_table(d, s.reference, link, s.citation)
                if len(norm):
                    collected.append(norm)
        except Exception:
            pass

    if not collected:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    mat = pd.concat(collected, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return mat


def save_outputs(matrix: pd.DataFrame, base_name: str = "graphene_polymer_literature_matrix"):
    matrix.to_csv(f"{base_name}.csv", index=False)
    matrix.to_excel(f"{base_name}.xlsx", index=False)
    print(f"Saved: {base_name}.csv and {base_name}.xlsx")


if __name__ == "__main__":
    sources = search_crossref_sources(year_from=1996, year_to=2026, rows=80)
    print(f"Discovered sources: {len(sources)}")
    matrix = build_literature_matrix(sources)
    print("Extracted rows:", len(matrix))
    print(matrix.head())
    save_outputs(matrix)
