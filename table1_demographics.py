from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class Table1Outputs:
    csv_path: Path
    md_path: Path
    tex_path: Path
    docx_path: Path | None = None


def _fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    if math.isnan(sd):
        return f"{mean:.1f}"
    return f"{mean:.1f} ({sd:.1f})"


def _fmt_median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    q1 = float(x.quantile(0.25))
    med = float(x.quantile(0.50))
    q3 = float(x.quantile(0.75))
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def _fmt_n_pct(mask: pd.Series) -> str:
    mask = mask.fillna(False).astype(bool)
    n = int(mask.sum())
    denom = int(len(mask))
    if denom == 0:
        return "0 (0.0%)"
    return f"{n} ({(100.0 * n / denom):.1f}%)"


def build_table1_demographics(
    demographic_csv: str | Path = "data/demographic.csv",
    *,
    analysis_csv: str | Path | None = "data/scr_amg_hipp_all_noShock.csv",
    analysis_subject_col: str = "sub",
    id_col: str = "sub_id",
    group_col: str = "group",
    age_col: str = "Age",
    gender_col: str = "Gender",
    gender_map: Mapping[Any, str] | None = None,
    group_label_map: Mapping[Any, str] | None = None,
    group_order: Sequence[str] | None = None,
    output_dir: str | Path = "outputs",
    filename_stem: str = "table1_demographics",
) -> tuple[pd.DataFrame, Table1Outputs]:
    """
    Build a "Table 1" demographics summary (overall + by group).

    If `analysis_csv` is provided, the demographics file is filtered to only the
    subjects present in that analysis dataset (mirrors `analysis_script.py`,
    where `sub_idx` is defined as categorical codes of the `sub` column).

    Exports:
      - CSV with rows as characteristics and columns as groups
      - Markdown table
      - LaTeX table (tabular)
    """
    demographic_csv = Path(demographic_csv)
    analysis_csv = Path(analysis_csv) if analysis_csv is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(demographic_csv)
    df = df.copy()

    if analysis_csv is not None:
        analysis_df = pd.read_csv(analysis_csv, usecols=[analysis_subject_col])
        allowed_subjects = (
            pd.Series(analysis_df[analysis_subject_col])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if id_col not in df.columns:
            raise KeyError(
                f"Cannot filter demographics by analysis subjects because `{id_col}` "
                f"was not found in {demographic_csv}."
            )
        df = df[df[id_col].astype(str).isin(set(allowed_subjects))].copy()

    if gender_map is None:
        # Common coding in many datasets is 1=Male, 2=Female.
        gender_map = {1: "Male", 2: "Female", 1.0: "Male", 2.0: "Female"}

    if group_label_map is None:
        group_label_map = {"VCC": "Combat Controls", "VPTSD": "PTSD", "HC": "Healthy Controls"}

    # Normalize types
    if age_col in df.columns:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    if gender_col in df.columns:
        # Keep original codes too; create a label column for clean display.
        df["_gender_label"] = df[gender_col].map(gender_map).fillna(df[gender_col].astype(str))
    else:
        df["_gender_label"] = pd.NA

    # Create a clean display label for group headers
    if group_col in df.columns:
        df["_group_label"] = df[group_col].map(group_label_map).fillna(df[group_col].astype(str))
    else:
        df["_group_label"] = pd.NA

    # Column order: Overall first, then groups (preferred order if provided)
    observed = list(pd.Series(df["_group_label"].dropna().unique()).astype(str))
    if group_order is None:
        preferred = ["Healthy Controls", "Combat Controls", "PTSD"]
        group_values = [g for g in preferred if g in observed] + sorted([g for g in observed if g not in preferred])
    else:
        # Keep only those present, append any other observed at end
        group_values = [g for g in group_order if g in observed] + [g for g in observed if g not in set(group_order)]

    col_order = ["Overall", *group_values]

    def subset_for(col_name: str) -> pd.DataFrame:
        if col_name == "Overall":
            return df
        return df[df["_group_label"].astype(str) == col_name]

    rows: dict[str, dict[str, str]] = {}

    for col_name in col_order:
        sdf = subset_for(col_name)

        # N (unique subjects if possible)
        if id_col in sdf.columns:
            n = int(sdf[id_col].nunique(dropna=True))
        else:
            n = int(len(sdf))

        rows.setdefault("N", {})[col_name] = str(n)
        rows.setdefault("Age, mean (SD)", {})[col_name] = _fmt_mean_sd(sdf.get(age_col, pd.Series(dtype=float)))
        rows.setdefault("Age, median [IQR]", {})[col_name] = _fmt_median_iqr(sdf.get(age_col, pd.Series(dtype=float)))

        # Gender distribution (based on label)
        gender_labels = sdf["_gender_label"]
        unique_labels = [x for x in pd.Series(gender_labels.dropna().unique()).astype(str) if x != "nan"]

        # Prefer a stable ordering if labels are Male/Female, else alphabetical.
        preferred = ["Female", "Male"]
        if set(preferred).issubset(set(unique_labels)):
            ordered = preferred + sorted([x for x in unique_labels if x not in preferred])
        else:
            ordered = sorted(unique_labels)

        for lab in ordered:
            rows.setdefault(f"Gender: {lab}, n (%)", {})[col_name] = _fmt_n_pct(gender_labels.astype(str) == lab)

    table = pd.DataFrame(rows).T.reindex(columns=col_order)
    table.index.name = "Characteristic"

    csv_path = output_dir / f"{filename_stem}.csv"
    md_path = output_dir / f"{filename_stem}.md"
    tex_path = output_dir / f"{filename_stem}.tex"
    docx_path = output_dir / f"{filename_stem}.docx"

    table.to_csv(csv_path)
    md_path.write_text(table.to_markdown(), encoding="utf-8")
    tex_path.write_text(table.to_latex(escape=True), encoding="utf-8")

    # Optional Word export (requires `python-docx`).
    wrote_docx = False
    try:
        from docx import Document  # type: ignore

        doc = Document()
        doc.add_heading("Table 1. Demographics", level=1)

        t = doc.add_table(rows=1, cols=len(table.columns) + 1)
        t.style = "Table Grid"

        hdr = t.rows[0].cells
        hdr[0].text = table.index.name or "Characteristic"
        for j, col in enumerate(table.columns, start=1):
            hdr[j].text = str(col)

        for idx, row in table.iterrows():
            cells = t.add_row().cells
            cells[0].text = str(idx)
            for j, col in enumerate(table.columns, start=1):
                val = row[col]
                cells[j].text = "" if pd.isna(val) else str(val)

        doc.save(docx_path)
        wrote_docx = True
    except Exception:
        wrote_docx = False

    return table, Table1Outputs(
        csv_path=csv_path,
        md_path=md_path,
        tex_path=tex_path,
        docx_path=(docx_path if wrote_docx else None),
    )


if __name__ == "__main__":
    table, paths = build_table1_demographics()
    print(table)
    print("\nWrote:")
    print(f"- {paths.csv_path}")
    print(f"- {paths.md_path}")
    print(f"- {paths.tex_path}")
    if paths.docx_path is not None:
        print(f"- {paths.docx_path}")
    else:
        print("- (DOCX not written; install `python-docx` to enable Word export)")

