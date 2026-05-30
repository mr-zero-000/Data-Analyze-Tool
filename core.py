"""
data_wrangling/core.py
======================
Core module for the DataWranglerTool package.

Provides two primary classes:
  - DataWrangler  : intelligent data loading, profiling, cleaning, reshaping,
                    feature engineering, and statistical insight.
  - ChartBuilder  : standalone, composable Plotly visualisations that return
                    self-contained HTML strings so they render in any notebook
                    or Colab cell.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily so the package installs even
# without them; explicit errors guide the user when they are missing.
# ---------------------------------------------------------------------------
def _require(pkg: str, extra: str = ""):
    try:
        return __import__(pkg)
    except ImportError:
        hint = f'  pip install data-wrangling-tool[{extra}]' if extra else f'  pip install {pkg}'
        raise ImportError(f"'{pkg}' is required for this feature.\n{hint}")


# ============================================================================
# Internal helpers
# ============================================================================

_NULL_STRINGS: List[str] = [
    "?", "??", "N/A", "n/a", "NA", "na", "NULL", "null", "Null",
    "None", "none", "NaN", "nan", "-", "--", "missing", "MISSING", "",
]

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns to numeric where possible."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace if more than 80 % of non-null values converted successfully
        non_null = df[col].notna().sum()
        if non_null > 0 and (converted.notna().sum() / non_null) >= 0.8:
            df[col] = converted
    return df


def _iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association strength between two categorical series."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _eta_squared(numeric: pd.Series, categorical: pd.Series) -> float:
    """η² (eta-squared) between a numeric and a categorical variable."""
    groups = [numeric[categorical == cat].dropna().values
               for cat in categorical.unique()]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    from scipy.stats import f_oneway
    _, p = f_oneway(*groups)
    grand_mean = numeric.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((numeric - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(classes="dw-table", border=0, index=True)


def _colab_display(html: str):
    display(HTML(html))


# ============================================================================
# DataWrangler
# ============================================================================

class DataWrangler:
    """
    A smart, interactive data wrangling toolkit for Google Colab.

    Workflow
    --------
    1. Load data          → ``upload_data()`` / ``load_url()`` / set ``self.df``
    2. Profile            → ``get_profile()`` / ``show_schema()`` / ``column_report()``
    3. Clean              → ``fix_nulls()`` / ``drop_duplicates()`` / ``fix_outliers()``
    4. Reshape            → ``rename_columns()`` / ``drop_columns()`` / ``drop_rows()``
                            ``cast_columns()`` / ``reorder_columns()``
    5. Engineer features  → ``bin_column()`` / ``extract_datetime_parts()``
                            ``add_ratio_column()`` / ``flag_outliers()``
    6. Encode & Scale     → ``encode_categoricals()`` / ``scale_numerics()``
    7. Visualise          → ``plot_numeric()`` / ``plot_categorical()``
                            ``plot_pair()`` / ``plot_timeseries()``
    8. Correlate          → ``plot_numeric_heatmap()`` / ``plot_categorical_heatmap()``
                            ``plot_unified_heatmap()``
    9. Export             → ``export_csv()`` / ``export_excel()`` / ``snapshot()``
    """

    # ------------------------------------------------------------------
    # Construction & state
    # ------------------------------------------------------------------

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self._history: List[str] = []           # human-readable operation log
        self._snapshots: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # ① DATA LOADING
    # ------------------------------------------------------------------

    def upload_data(self, null_strings: Optional[List[str]] = None):
        """
        Interactive file upload for Google Colab.
        Accepts CSV, TSV, XLSX / XLS, JSON, and Parquet.
        """
        try:
            from google.colab import files as colab_files
        except ImportError:
            raise EnvironmentError(
                "upload_data() requires Google Colab. "
                "Outside Colab, assign a DataFrame directly: wrangler.df = pd.read_csv('...')"
            )

        nulls = null_strings or _NULL_STRINGS
        uploaded = colab_files.upload()
        for fname, data in uploaded.items():
            ext = fname.rsplit(".", 1)[-1].lower()
            buf = io.BytesIO(data)
            if ext == "csv":
                self.df = pd.read_csv(buf, na_values=nulls, keep_default_na=True)
            elif ext == "tsv":
                self.df = pd.read_csv(buf, sep="\t", na_values=nulls, keep_default_na=True)
            elif ext in ("xlsx", "xls"):
                self.df = pd.read_excel(buf, na_values=nulls)
            elif ext == "json":
                self.df = pd.read_json(buf)
            elif ext == "parquet":
                self.df = pd.read_parquet(buf)
            else:
                raise ValueError(f"Unsupported file type: .{ext}")

            self.df = _coerce_numeric(self.df)
            self._log(f"Uploaded '{fname}' → shape {self.df.shape}")
            print(f"✅ Loaded '{fname}'  |  {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            break  # process only the first file

    def load_url(
        self,
        url: str,
        file_type: Literal["csv", "tsv", "excel", "json", "parquet"] = "csv",
        null_strings: Optional[List[str]] = None,
        **kwargs,
    ):
        """Load data directly from a URL."""
        nulls = null_strings or _NULL_STRINGS
        readers = {
            "csv":     lambda: pd.read_csv(url, na_values=nulls, keep_default_na=True, **kwargs),
            "tsv":     lambda: pd.read_csv(url, sep="\t", na_values=nulls, keep_default_na=True, **kwargs),
            "excel":   lambda: pd.read_excel(url, na_values=nulls, **kwargs),
            "json":    lambda: pd.read_json(url, **kwargs),
            "parquet": lambda: pd.read_parquet(url, **kwargs),
        }
        if file_type not in readers:
            raise ValueError(f"file_type must be one of {list(readers)}")
        self.df = _coerce_numeric(readers[file_type]())
        self._log(f"Loaded URL '{url}' → shape {self.df.shape}")
        print(f"✅ Loaded from URL  |  {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")

    # ------------------------------------------------------------------
    # ② PROFILING & INSPECTION
    # ------------------------------------------------------------------

    def get_profile(self):
        """Print a rich summary: shape, dtypes breakdown, memory, missing %."""
        self._check_df()
        df = self.df
        n, m = df.shape
        mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        dt_cols  = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        bool_cols = df.select_dtypes(include="bool").columns.tolist()

        lines = [
            "━" * 55,
            f"  DataWrangler Profile",
            "━" * 55,
            f"  Rows            : {n:,}",
            f"  Columns         : {m}",
            f"  Numeric cols    : {len(num_cols)}",
            f"  Categorical cols: {len(cat_cols)}",
            f"  Datetime cols   : {len(dt_cols)}",
            f"  Boolean cols    : {len(bool_cols)}",
            f"  Memory usage    : {mem:.2f} MB",
            f"  Duplicate rows  : {df.duplicated().sum():,}",
            "━" * 55,
        ]
        print("\n".join(lines))

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing):
            print("\n  Missing Values:")
            pct = (missing / n * 100).round(2)
            miss_df = pd.DataFrame({"Missing": missing, "Pct (%)": pct})
            _colab_display(_html_table(miss_df))
        else:
            print("\n  ✅ No missing values found.")

    def show_schema(self):
        """Tabular view of column names, dtypes, unique counts, and sample values."""
        self._check_df()
        df = self.df
        rows = []
        for col in df.columns:
            rows.append({
                "Column": col,
                "Dtype": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null": df[col].isna().sum(),
                "Unique": df[col].nunique(),
                "Sample Values": str(df[col].dropna().unique()[:3].tolist()),
            })
        schema = pd.DataFrame(rows)
        _colab_display(_html_table(schema))

    def column_report(self, column: str):
        """
        Deep report for a single column:
        - Numeric  → count, mean, std, min/max, quartiles, skewness, kurtosis, outlier count
        - Categorical → top-10 value counts with bar visualisation
        - Datetime → min, max, range, gaps
        """
        self._check_df()
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found.")
        series = self.df[column]
        print(f"\n── Column Report: '{column}' ({series.dtype}) ──")
        print(f"  Non-null : {series.notna().sum():,}  |  Null : {series.isna().sum():,}")
        print(f"  Unique   : {series.nunique():,}")

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            lo, hi = _iqr_bounds(series.dropna())
            outliers = ((series < lo) | (series > hi)).sum()
            print(f"\n  Numeric stats:")
            for k, v in desc.items():
                print(f"    {k:8s} : {v:.4g}")
            print(f"    Skewness : {series.skew():.4g}")
            print(f"    Kurtosis : {series.kurtosis():.4g}")
            print(f"    Outliers (IQR×1.5): {outliers:,}")
        elif pd.api.types.is_datetime64_any_dtype(series):
            print(f"\n  Datetime range: {series.min()} → {series.max()}")
            print(f"  Range span    : {series.max() - series.min()}")
        else:
            vc = series.value_counts().head(10)
            print(f"\n  Top values:")
            for val, cnt in vc.items():
                bar = "█" * int(cnt / vc.max() * 20)
                print(f"    {str(val):<25} {cnt:>6,}  {bar}")

    def get_numeric_summary(self):
        """Extended descriptive statistics for all numeric columns, including skewness & kurtosis."""
        self._check_df()
        num = self.df.select_dtypes(include="number")
        if num.empty:
            print("No numeric columns found.")
            return
        desc = num.describe().T
        desc["skewness"] = num.skew()
        desc["kurtosis"] = num.kurtosis()
        _colab_display(_html_table(desc.round(4)))

    def get_categorical_summary(self):
        """Value counts & entropy for all categorical columns."""
        self._check_df()
        cats = self.df.select_dtypes(include="object")
        if cats.empty:
            print("No categorical columns found.")
            return
        from scipy.stats import entropy as sp_entropy
        rows = []
        for col in cats.columns:
            vc = cats[col].value_counts(normalize=True)
            rows.append({
                "Column": col,
                "Unique": cats[col].nunique(),
                "Top Value": vc.index[0] if len(vc) else "—",
                "Top Freq (%)": round(vc.iloc[0] * 100, 2) if len(vc) else 0,
                "Shannon Entropy": round(sp_entropy(vc.values), 4),
            })
        _colab_display(_html_table(pd.DataFrame(rows).set_index("Column")))

    def show_missing_heatmap(self):
        """Visualise the missingness pattern across columns as a heatmap."""
        self._check_df()
        px = _require("plotly.express", "plotting")
        miss = self.df.isnull().astype(int)
        fig = px.imshow(
            miss.T,
            color_continuous_scale=["#e8f4f8", "#c44e52"],
            labels={"color": "Missing"},
            title="Missing-Value Map (1 = missing)",
            aspect="auto",
        )
        fig.update_layout(height=max(300, 25 * len(self.df.columns)))
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    # ------------------------------------------------------------------
    # ③ CLEANING
    # ------------------------------------------------------------------

    def fix_nulls(
        self,
        strategy: Literal["mean", "median", "mode", "constant", "ffill", "bfill", "drop"] = "median",
        fill_value: Any = 0,
        columns: Optional[List[str]] = None,
    ):
        """
        Impute or remove missing values.

        Parameters
        ----------
        strategy   : 'mean' | 'median' | 'mode' | 'constant' | 'ffill' | 'bfill' | 'drop'
        fill_value : Used only with strategy='constant'
        columns    : Restrict to these columns; defaults to all columns with nulls
        """
        self._check_df()
        cols = columns or self.df.columns[self.df.isnull().any()].tolist()
        before = self.df.isnull().sum().sum()

        for col in cols:
            if col not in self.df.columns:
                warnings.warn(f"Column '{col}' not found, skipped.")
                continue
            s = self.df[col]
            if strategy == "mean":
                self.df[col] = s.fillna(s.mean() if pd.api.types.is_numeric_dtype(s) else s.mode()[0])
            elif strategy == "median":
                self.df[col] = s.fillna(s.median() if pd.api.types.is_numeric_dtype(s) else s.mode()[0])
            elif strategy == "mode":
                self.df[col] = s.fillna(s.mode()[0] if not s.mode().empty else fill_value)
            elif strategy == "constant":
                self.df[col] = s.fillna(fill_value)
            elif strategy == "ffill":
                self.df[col] = s.ffill()
            elif strategy == "bfill":
                self.df[col] = s.bfill()
            elif strategy == "drop":
                self.df = self.df.dropna(subset=[col])

        after = self.df.isnull().sum().sum()
        self._log(f"fix_nulls(strategy='{strategy}') → fixed {before - after} null(s)")
        print(f"✅ Nulls fixed: {before - after} cells resolved  (strategy='{strategy}')")

    def drop_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first"):
        """Remove exact duplicate rows."""
        self._check_df()
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        removed = before - len(self.df)
        self._log(f"drop_duplicates → removed {removed} rows")
        print(f"✅ Removed {removed:,} duplicate row(s)  |  Remaining: {len(self.df):,}")

    def fix_outliers(
        self,
        columns: Optional[List[str]] = None,
        strategy: Literal["remove", "clip", "winsorise"] = "clip",
        k: float = 1.5,
    ):
        """
        Detect and handle outliers using the IQR method.

        Parameters
        ----------
        columns  : Numeric columns to inspect; defaults to all
        strategy : 'remove' drops rows, 'clip' / 'winsorise' cap values at IQR bounds
        k        : IQR multiplier (default 1.5)
        """
        self._check_df()
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in (columns or num_cols) if c in num_cols]
        before = len(self.df)

        if strategy == "remove":
            mask = pd.Series([True] * len(self.df), index=self.df.index)
            for col in cols:
                lo, hi = _iqr_bounds(self.df[col].dropna(), k)
                mask &= self.df[col].between(lo, hi)
            self.df = self.df[mask].reset_index(drop=True)
        else:  # clip or winsorise
            for col in cols:
                lo, hi = _iqr_bounds(self.df[col].dropna(), k)
                self.df[col] = self.df[col].clip(lower=lo, upper=hi)

        affected = before - len(self.df) if strategy == "remove" else 0
        self._log(f"fix_outliers(strategy='{strategy}', k={k}) on {cols}")
        print(f"✅ Outliers handled via '{strategy}'  |  Rows removed: {affected:,}")

    def flag_outliers(self, columns: Optional[List[str]] = None, k: float = 1.5) -> pd.DataFrame:
        """
        Return a boolean DataFrame marking outlier cells (True = outlier).
        Does NOT modify self.df.
        """
        self._check_df()
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in (columns or num_cols) if c in num_cols]
        flags = pd.DataFrame(False, index=self.df.index, columns=cols)
        for col in cols:
            lo, hi = _iqr_bounds(self.df[col].dropna(), k)
            flags[col] = (self.df[col] < lo) | (self.df[col] > hi)
        print(f"ℹ️  Outlier flags computed for {len(cols)} column(s).  Use `.sum()` to count per column.")
        return flags

    # ------------------------------------------------------------------
    # ④ RESHAPING
    # ------------------------------------------------------------------

    def drop_columns(self, columns: Optional[List[str]] = None):
        """
        Drop columns interactively (if no list given) or by name.
        Interactive mode prints a numbered list and prompts for input.
        """
        self._check_df()
        if columns is None:
            print("Available columns:")
            for i, c in enumerate(self.df.columns):
                print(f"  [{i}] {c}")
            raw = input("\nEnter column indices or names (comma-separated) to drop: ")
            tokens = [t.strip() for t in raw.split(",")]
            columns = []
            for t in tokens:
                if t.isdigit() and int(t) < len(self.df.columns):
                    columns.append(self.df.columns[int(t)])
                elif t in self.df.columns:
                    columns.append(t)
                else:
                    print(f"  ⚠️  '{t}' not recognised, skipped.")

        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            warnings.warn(f"Columns not found and skipped: {missing}")
        to_drop = [c for c in columns if c in self.df.columns]
        self.df = self.df.drop(columns=to_drop)
        self._log(f"Dropped columns: {to_drop}")
        print(f"✅ Dropped {len(to_drop)} column(s): {to_drop}")

    def drop_rows(self, indices: Optional[List[int]] = None):
        """
        Drop rows by index. If no list is given, shows the head and prompts.
        """
        self._check_df()
        if indices is None:
            print(self.df.head(20).to_string())
            raw = input("\nEnter row index values to drop (comma-separated): ")
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]

        valid = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=valid).reset_index(drop=True)
        self._log(f"Dropped rows at indices: {valid}")
        print(f"✅ Dropped {len(valid)} row(s)")

    def rename_columns(self, mapping: Optional[Dict[str, str]] = None):
        """
        Rename columns. If ``mapping`` is None, interactive prompt is shown.

        Parameters
        ----------
        mapping : dict like ``{'old_name': 'new_name', ...}``
        """
        self._check_df()
        if mapping is None:
            print("Current columns:", list(self.df.columns))
            raw = input("Enter renaming pairs as old:new (comma-separated, e.g. age:Age, name:Name): ")
            mapping = {}
            for pair in raw.split(","):
                if ":" in pair:
                    old, new = pair.split(":", 1)
                    mapping[old.strip()] = new.strip()

        self.df = self.df.rename(columns=mapping)
        self._log(f"Renamed columns: {mapping}")
        print(f"✅ Renamed {len(mapping)} column(s)")

    def cast_columns(self, mapping: Dict[str, str]):
        """
        Cast columns to specified dtypes.

        Parameters
        ----------
        mapping : dict like ``{'col': 'float', 'date_col': 'datetime', ...}``
        """
        self._check_df()
        for col, dtype in mapping.items():
            if col not in self.df.columns:
                warnings.warn(f"Column '{col}' not found, skipped.")
                continue
            try:
                if dtype in ("datetime", "datetime64"):
                    self.df[col] = pd.to_datetime(self.df[col], infer_datetime_format=True)
                else:
                    self.df[col] = self.df[col].astype(dtype)
            except Exception as e:
                warnings.warn(f"Could not cast '{col}' to '{dtype}': {e}")
        self._log(f"cast_columns: {mapping}")
        print(f"✅ Casted {len(mapping)} column(s)")

    def reorder_columns(self, columns: List[str]):
        """Reorder the DataFrame columns. Unlisted columns are appended at the end."""
        self._check_df()
        rest = [c for c in self.df.columns if c not in columns]
        self.df = self.df[columns + rest]
        self._log(f"Reordered columns to: {columns} + rest")
        print(f"✅ Columns reordered")

    # ------------------------------------------------------------------
    # ⑤ FEATURE ENGINEERING
    # ------------------------------------------------------------------

    def bin_column(
        self,
        column: str,
        bins: Union[int, List[float]] = 5,
        labels: Optional[List[str]] = None,
        new_column: Optional[str] = None,
    ):
        """
        Bin a numeric column into discrete categories.

        Parameters
        ----------
        column     : Source numeric column name
        bins       : Number of equal-width bins or explicit bin edges
        labels     : Custom bin labels
        new_column : Output column name (defaults to ``<column>_bin``)
        """
        self._check_df()
        out = new_column or f"{column}_bin"
        self.df[out] = pd.cut(self.df[column], bins=bins, labels=labels, include_lowest=True)
        self._log(f"bin_column('{column}') → '{out}'")
        print(f"✅ Binned '{column}' → '{out}'  |  {self.df[out].value_counts().to_dict()}")

    def extract_datetime_parts(self, column: str, parts: Optional[List[str]] = None):
        """
        Expand a datetime column into component columns.

        Parameters
        ----------
        column : Datetime column name
        parts  : Subset of ['year','month','day','hour','minute','second',
                             'dayofweek','weekday_name','quarter','week']
        """
        self._check_df()
        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            self.df[column] = pd.to_datetime(self.df[column], infer_datetime_format=True)
        dt = self.df[column].dt
        all_parts = {
            "year": dt.year, "month": dt.month, "day": dt.day,
            "hour": dt.hour, "minute": dt.minute, "second": dt.second,
            "dayofweek": dt.dayofweek,
            "weekday_name": dt.day_name(),
            "quarter": dt.quarter,
            "week": dt.isocalendar().week.astype(int),
        }
        selected = parts or list(all_parts.keys())
        new_cols = []
        for p in selected:
            if p in all_parts:
                col_name = f"{column}_{p}"
                self.df[col_name] = all_parts[p]
                new_cols.append(col_name)
        self._log(f"extract_datetime_parts('{column}') → {new_cols}")
        print(f"✅ Extracted {len(new_cols)} part(s) from '{column}': {new_cols}")

    def add_ratio_column(self, numerator: str, denominator: str, new_column: Optional[str] = None):
        """Create a new column = numerator / denominator (division by zero → NaN)."""
        self._check_df()
        out = new_column or f"{numerator}_per_{denominator}"
        self.df[out] = self.df[numerator] / self.df[denominator].replace(0, np.nan)
        self._log(f"add_ratio_column('{numerator}'/'{denominator}') → '{out}'")
        print(f"✅ Created ratio column '{out}'")

    def apply_transform(self, column: str, func, new_column: Optional[str] = None):
        """
        Apply an arbitrary function to a column.

        Parameters
        ----------
        column     : Source column
        func       : Callable applied element-wise (e.g. ``np.log1p``)
        new_column : Output column (defaults to ``<column>_transformed``)
        """
        self._check_df()
        out = new_column or f"{column}_transformed"
        self.df[out] = self.df[column].apply(func)
        self._log(f"apply_transform('{column}') → '{out}'")
        print(f"✅ Transformed '{column}' → '{out}'")

    # ------------------------------------------------------------------
    # ⑥ ENCODING & SCALING
    # ------------------------------------------------------------------

    def encode_categoricals(
        self,
        method: Literal["onehot", "ordinal", "label", "frequency"] = "onehot",
        columns: Optional[List[str]] = None,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Returns the encoded DataFrame and updates ``self.df`` if ``drop_original=True``.
        """
        self._check_df()
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = [c for c in (columns or cat_cols) if c in self.df.columns]

        if method == "onehot":
            encoded = pd.get_dummies(self.df[cols], prefix=cols, drop_first=False)
            if drop_original:
                self.df = pd.concat([self.df.drop(columns=cols), encoded], axis=1)
        elif method in ("ordinal", "label"):
            from sklearn.preprocessing import OrdinalEncoder
            enc = OrdinalEncoder()
            self.df[[f"{c}_enc" for c in cols]] = enc.fit_transform(self.df[cols].astype(str))
            if drop_original:
                self.df = self.df.drop(columns=cols)
        elif method == "frequency":
            for c in cols:
                freq_map = self.df[c].value_counts(normalize=True)
                self.df[f"{c}_freq"] = self.df[c].map(freq_map)
                if drop_original:
                    self.df = self.df.drop(columns=[c])

        self._log(f"encode_categoricals(method='{method}', columns={cols})")
        print(f"✅ Encoded {len(cols)} categorical column(s) via '{method}'")
        return self.df

    def scale_numerics(
        self,
        method: Literal["minmax", "standard", "robust", "log1p", "sqrt"] = "standard",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Scale numeric columns in-place.

        Parameters
        ----------
        method  : 'minmax' | 'standard' (z-score) | 'robust' | 'log1p' | 'sqrt'
        columns : Columns to scale; defaults to all numeric columns
        """
        self._check_df()
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in (columns or num_cols) if c in self.df.columns]

        if method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.df[cols] = MinMaxScaler().fit_transform(self.df[cols])
        elif method == "standard":
            from sklearn.preprocessing import StandardScaler
            self.df[cols] = StandardScaler().fit_transform(self.df[cols])
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            self.df[cols] = RobustScaler().fit_transform(self.df[cols])
        elif method == "log1p":
            for c in cols:
                if (self.df[c] >= 0).all():
                    self.df[c] = np.log1p(self.df[c])
                else:
                    warnings.warn(f"log1p skipped for '{c}': contains negative values.")
        elif method == "sqrt":
            for c in cols:
                if (self.df[c] >= 0).all():
                    self.df[c] = np.sqrt(self.df[c])
                else:
                    warnings.warn(f"sqrt skipped for '{c}': contains negative values.")

        self._log(f"scale_numerics(method='{method}', columns={cols})")
        print(f"✅ Scaled {len(cols)} numeric column(s) via '{method}'")
        return self.df

    def get_encoded_df(self, numeric_method: str = "standard", categorical_method: str = "onehot") -> pd.DataFrame:
        """
        Convenience: returns a fully encoded + scaled copy of the DataFrame
        ready for ML training, without modifying ``self.df``.
        """
        self._check_df()
        import copy
        original = self.df
        self.df = self.df.copy()
        self.encode_categoricals(method=categorical_method)
        self.scale_numerics(method=numeric_method)
        result = self.df.copy()
        self.df = original
        print(f"✅ ML-ready DataFrame: {result.shape[0]:,} rows × {result.shape[1]} columns")
        return result

    # ------------------------------------------------------------------
    # ⑦ VISUALISATIONS
    # ------------------------------------------------------------------

    def plot_numeric(self, column_names: Optional[List[str]] = None):
        """
        For each numeric column render a three-panel view:
        violin (distribution shape) + box-plot strip + histogram.
        """
        self._check_df()
        go = _require("plotly.graph_objects", "plotting")
        from plotly.subplots import make_subplots
        import plotly.express as px

        cols = column_names or self.df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]
        if not cols:
            print("No numeric columns to plot.")
            return

        for col in cols:
            data = self.df[col].dropna()
            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=["Violin", "Box Plot", "Histogram"])
            fig.add_trace(go.Violin(y=data, name=col, fillcolor=_PALETTE[0],
                                    line_color="white", box_visible=True, meanline_visible=True,
                                    opacity=0.75), row=1, col=1)
            fig.add_trace(go.Box(y=data, name=col, marker_color=_PALETTE[1],
                                 boxpoints="outliers"), row=1, col=2)
            fig.add_trace(go.Histogram(x=data, name=col, marker_color=_PALETTE[2],
                                       opacity=0.8, nbinsx=30), row=1, col=3)
            fig.update_layout(title_text=f"Distribution of '{col}'",
                               template="plotly_dark", showlegend=False, height=380)
            _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_categorical(self, column_names: Optional[List[str]] = None, top_n: int = 15):
        """
        Horizontal bar chart of value frequencies for each categorical column.
        """
        self._check_df()
        import plotly.express as px
        cols = column_names or self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cols = [c for c in cols if c in self.df.columns]
        if not cols:
            print("No categorical columns to plot.")
            return
        for col in cols:
            vc = self.df[col].value_counts().head(top_n).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x="count", y=col, orientation="h",
                         title=f"Top {top_n} values in '{col}'",
                         color_discrete_sequence=[_PALETTE[3]],
                         template="plotly_dark")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(350, 28 * len(vc)))
            _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_pair(self, columns: Optional[List[str]] = None, color: Optional[str] = None):
        """
        Scatter-matrix (pair plot) for numeric columns.

        Parameters
        ----------
        columns : Numeric columns to include
        color   : Optional categorical column to colour the points by
        """
        self._check_df()
        import plotly.express as px
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in (columns or num_cols) if c in num_cols][:8]  # cap at 8 for readability
        if len(cols) < 2:
            print("Need at least 2 numeric columns for a pair plot.")
            return
        fig = px.scatter_matrix(self.df, dimensions=cols, color=color,
                                title="Pair Plot",
                                template="plotly_dark",
                                color_discrete_sequence=_PALETTE)
        fig.update_traces(diagonal_visible=False, marker_size=3)
        fig.update_layout(height=700)
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_relationship(self, col_a: str, col_b: str, color: Optional[str] = None):
        """
        Auto-select the most appropriate chart for two columns:
        - Numeric × Numeric    → scatter with trend-line
        - Categorical × Numeric → box plot per category
        - Categorical × Categorical → grouped bar chart
        """
        self._check_df()
        import plotly.express as px
        a_num = pd.api.types.is_numeric_dtype(self.df[col_a])
        b_num = pd.api.types.is_numeric_dtype(self.df[col_b])

        if a_num and b_num:
            fig = px.scatter(self.df, x=col_a, y=col_b, color=color,
                             trendline="ols",
                             title=f"Scatter: {col_a} vs {col_b}",
                             template="plotly_dark",
                             color_discrete_sequence=_PALETTE)
        elif (not a_num) and b_num:
            fig = px.box(self.df, x=col_a, y=col_b, color=color,
                         title=f"Box: {col_b} by {col_a}",
                         template="plotly_dark",
                         color_discrete_sequence=_PALETTE)
        elif a_num and (not b_num):
            fig = px.box(self.df, x=col_b, y=col_a, color=color,
                         title=f"Box: {col_a} by {col_b}",
                         template="plotly_dark",
                         color_discrete_sequence=_PALETTE)
        else:
            ct = pd.crosstab(self.df[col_a], self.df[col_b]).reset_index().melt(id_vars=col_a)
            fig = px.bar(ct, x=col_a, y="value", color=col_b, barmode="group",
                         title=f"Grouped Bar: {col_a} × {col_b}",
                         template="plotly_dark",
                         color_discrete_sequence=_PALETTE)
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_timeseries(
        self,
        time_col: str,
        value_cols: Optional[List[str]] = None,
        resample_rule: Optional[str] = None,
        agg: str = "mean",
    ):
        """
        Line chart for time-series data.

        Parameters
        ----------
        time_col      : Datetime column
        value_cols    : Numeric columns to plot
        resample_rule : Pandas offset alias e.g. 'D', 'W', 'M'
        agg           : Aggregation method when resampling
        """
        self._check_df()
        import plotly.express as px
        df = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in (value_cols or num_cols) if c in num_cols]
        if resample_rule:
            df = getattr(df[cols].resample(resample_rule), agg)()
        else:
            df = df[cols]
        df = df.reset_index()
        fig = px.line(df, x=time_col, y=cols,
                      title=f"Time Series: {', '.join(cols)}",
                      template="plotly_dark",
                      color_discrete_sequence=_PALETTE)
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    # ------------------------------------------------------------------
    # ⑧ CORRELATION & ASSOCIATION
    # ------------------------------------------------------------------

    def plot_numeric_heatmap(self, method: Literal["pearson", "spearman", "kendall"] = "pearson"):
        """Correlation heatmap for numeric columns."""
        self._check_df()
        import plotly.express as px
        num = self.df.select_dtypes(include="number")
        if num.shape[1] < 2:
            print("Need at least 2 numeric columns.")
            return
        corr = num.corr(method=method).round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title=f"{method.capitalize()} Correlation Heatmap",
                        template="plotly_dark",
                        aspect="auto")
        fig.update_layout(height=max(400, 40 * len(corr)))
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_categorical_heatmap(self):
        """Cramér's V association heatmap for categorical columns."""
        self._check_df()
        import plotly.express as px
        cats = self.df.select_dtypes(include=["object", "category"])
        if cats.shape[1] < 2:
            print("Need at least 2 categorical columns.")
            return
        cols = cats.columns.tolist()
        matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i in cols:
            for j in cols:
                matrix.loc[i, j] = 1.0 if i == j else _cramers_v(cats[i].dropna(), cats[j].dropna())
        fig = px.imshow(matrix.astype(float).round(3), text_auto=True,
                        color_continuous_scale="Blues",
                        zmin=0, zmax=1,
                        title="Cramér's V Association Heatmap (Categorical)",
                        template="plotly_dark", aspect="auto")
        fig.update_layout(height=max(400, 40 * len(cols)))
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    def plot_unified_heatmap(self):
        """
        Unified association matrix combining:
        - Pearson r (numeric × numeric)
        - Cramér's V (categorical × categorical)
        - η² / Eta-squared (numeric × categorical)
        All values are normalised to [0, 1].
        """
        self._check_df()
        import plotly.express as px
        df = self.df.copy()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        all_cols = num_cols + cat_cols
        if len(all_cols) < 2:
            print("Not enough columns for a unified heatmap.")
            return

        n = len(all_cols)
        matrix = pd.DataFrame(np.nan, index=all_cols, columns=all_cols)

        for i, ci in enumerate(all_cols):
            for j, cj in enumerate(all_cols):
                if i == j:
                    matrix.loc[ci, cj] = 1.0
                    continue
                i_num = ci in num_cols
                j_num = cj in num_cols
                if i_num and j_num:
                    r = df[[ci, cj]].dropna().corr().iloc[0, 1]
                    matrix.loc[ci, cj] = abs(float(r))
                elif (not i_num) and (not j_num):
                    a = df[ci].dropna()
                    b = df[cj].dropna()
                    idx = a.index.intersection(b.index)
                    matrix.loc[ci, cj] = _cramers_v(a[idx], b[idx])
                else:
                    num_c, cat_c = (ci, cj) if i_num else (cj, ci)
                    sub = df[[num_c, cat_c]].dropna()
                    matrix.loc[ci, cj] = _eta_squared(sub[num_c], sub[cat_c])

        fig = px.imshow(matrix.astype(float).round(3), text_auto=True,
                        color_continuous_scale="Viridis",
                        zmin=0, zmax=1,
                        title="Unified Association Heatmap  (|Pearson| / Cramér's V / η²)",
                        template="plotly_dark", aspect="auto")
        fig.update_layout(height=max(500, 40 * n))
        _colab_display(fig.to_html(include_plotlyjs="cdn", full_html=False))

    # ------------------------------------------------------------------
    # ⑨ EXPORT & SNAPSHOT
    # ------------------------------------------------------------------

    def export_csv(self, path: str = "wrangled_data.csv", index: bool = False):
        """Save the current DataFrame to CSV."""
        self._check_df()
        self.df.to_csv(path, index=index)
        print(f"✅ Exported to '{path}'")

    def export_excel(self, path: str = "wrangled_data.xlsx", sheet: str = "Sheet1", index: bool = False):
        """Save the current DataFrame to Excel (.xlsx)."""
        self._check_df()
        self.df.to_excel(path, sheet_name=sheet, index=index)
        print(f"✅ Exported to '{path}'")

    def snapshot(self, name: str):
        """Save a named snapshot of the current DataFrame for rollback."""
        self._check_df()
        self._snapshots[name] = self.df.copy()
        print(f"📸 Snapshot '{name}' saved  |  shape: {self.df.shape}")

    def restore(self, name: str):
        """Restore a previously saved snapshot."""
        if name not in self._snapshots:
            raise KeyError(f"No snapshot named '{name}'.  Available: {list(self._snapshots)}")
        self.df = self._snapshots[name].copy()
        print(f"⏪ Restored snapshot '{name}'  |  shape: {self.df.shape}")

    def get_history(self):
        """Print the operation history log."""
        if not self._history:
            print("No operations recorded yet.")
        for i, entry in enumerate(self._history, 1):
            print(f"  {i:>3}. {entry}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_df(self):
        if self.df is None:
            raise RuntimeError("No data loaded. Call upload_data(), load_url(), or set wrangler.df directly.")

    def _log(self, msg: str):
        self._history.append(msg)


# ============================================================================
# ChartBuilder
# ============================================================================

class ChartBuilder:
    """
    A standalone collection of composable Plotly charts.

    Every method returns ``{'status': 'ok'|'error', 'html': '...', 'message': '...'}``
    and can be rendered with ``ChartBuilder.display(result)``.
    """

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    @staticmethod
    def display(result: Dict[str, str]):
        """Render the HTML output of any ChartBuilder method in the current cell."""
        if result.get("status") == "error":
            print(f"⚠️  Chart error: {result.get('message')}")
        else:
            display(HTML(result.get("html", "")))

    @staticmethod
    def get_methods_info() -> Dict[str, List[Dict]]:
        """Return a metadata table of all chart methods."""
        methods = [
            {"Method": "plot_bar", "Description": "Grouped / stacked bar chart",
             "Key Args": "x, y, color, barmode"},
            {"Method": "plot_pie", "Description": "Pie / donut chart",
             "Key Args": "names, values, hole"},
            {"Method": "plot_histogram", "Description": "Histogram with optional KDE",
             "Key Args": "x, bins, kde"},
            {"Method": "plot_scatter", "Description": "Scatter with optional regression line",
             "Key Args": "x, y, color, trendline"},
            {"Method": "plot_line", "Description": "Line chart (multi-series)",
             "Key Args": "x, y_cols"},
            {"Method": "plot_heatmap", "Description": "Pivot-aggregated heatmap",
             "Key Args": "index, columns, values, agg"},
            {"Method": "plot_sankey", "Description": "Sankey flow diagram",
             "Key Args": "source_col, target_col, value_col"},
            {"Method": "plot_sunburst", "Description": "Multi-level sunburst chart",
             "Key Args": "path, values"},
            {"Method": "plot_funnel", "Description": "Funnel chart",
             "Key Args": "x, y"},
            {"Method": "plot_treemap", "Description": "Treemap for hierarchical data",
             "Key Args": "path, values, color"},
            {"Method": "plot_bubble", "Description": "Bubble chart (scatter + size)",
             "Key Args": "x, y, size, color"},
        ]
        return {"response": methods}

    # ------------------------------------------------------------------
    # Chart methods
    # ------------------------------------------------------------------

    def _wrap(self, fig) -> Dict[str, str]:
        try:
            html = fig.to_html(include_plotlyjs="cdn", full_html=False)
            return {"status": "ok", "html": html}
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        barmode: str = "group",
        title: str = "",
        orientation: str = "v",
    ) -> Dict[str, str]:
        """Grouped / stacked bar chart."""
        try:
            import plotly.express as px
            fig = px.bar(data, x=x, y=y, color=color, barmode=barmode,
                         title=title or f"{y} by {x}",
                         orientation=orientation,
                         template="plotly_dark",
                         color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_pie(
        self,
        data: pd.DataFrame,
        names: str,
        values: str,
        hole: float = 0.0,
        title: str = "",
    ) -> Dict[str, str]:
        """Pie / donut chart.  Set ``hole > 0`` for a donut."""
        try:
            import plotly.express as px
            fig = px.pie(data, names=names, values=values, hole=hole,
                         title=title or f"{values} distribution by {names}",
                         template="plotly_dark",
                         color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_histogram(
        self,
        data: pd.DataFrame,
        x: str,
        bins: Optional[Union[int, List]] = None,
        kde: bool = False,
        title: str = "",
        color: Optional[str] = None,
    ) -> Dict[str, str]:
        """Histogram with optional marginal KDE rug."""
        try:
            import plotly.express as px
            marginal = "rug" if kde else None
            kw: Dict = dict(nbins=bins) if isinstance(bins, int) else {}
            fig = px.histogram(data, x=x, color=color, marginal=marginal,
                               title=title or f"Distribution of {x}",
                               template="plotly_dark",
                               color_discrete_sequence=_PALETTE, **kw)
            if isinstance(bins, list):
                fig.update_traces(xbins=dict(start=bins[0], end=bins[-1],
                                              size=(bins[-1] - bins[0]) / (len(bins) - 1)))
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        trendline: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, str]:
        """Scatter plot with optional regression trend-line."""
        try:
            import plotly.express as px
            fig = px.scatter(data, x=x, y=y, color=color, size=size,
                             trendline=trendline,
                             title=title or f"{x} vs {y}",
                             template="plotly_dark",
                             color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_line(
        self,
        data: pd.DataFrame,
        x: str,
        y_cols: Union[str, List[str]],
        title: str = "",
        markers: bool = False,
    ) -> Dict[str, str]:
        """Multi-series line chart."""
        try:
            import plotly.express as px
            cols = [y_cols] if isinstance(y_cols, str) else y_cols
            fig = px.line(data, x=x, y=cols,
                          title=title or f"Line chart: {', '.join(cols)}",
                          template="plotly_dark",
                          markers=markers,
                          color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_heatmap(
        self,
        data: pd.DataFrame,
        index: str,
        columns: str,
        values: str,
        agg: str = "mean",
        title: str = "",
    ) -> Dict[str, str]:
        """Pivot-aggregated heatmap."""
        try:
            import plotly.express as px
            pivot = data.pivot_table(index=index, columns=columns, values=values,
                                     aggfunc=agg).round(3)
            fig = px.imshow(pivot, text_auto=True,
                            color_continuous_scale="Viridis",
                            title=title or f"{agg}({values}) by {index} × {columns}",
                            template="plotly_dark", aspect="auto")
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_sankey(
        self,
        data: pd.DataFrame,
        source_col: str,
        target_col: str,
        value_col: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, str]:
        """Sankey flow diagram between two categorical columns."""
        try:
            import plotly.graph_objects as go
            agg = data.groupby([source_col, target_col])
            if value_col:
                flow = agg[value_col].sum().reset_index()
                flow.columns = ["source", "target", "value"]
            else:
                flow = agg.size().reset_index(name="value")

            labels = list(pd.concat([flow["source"], flow["target"]]).unique())
            label_idx = {l: i for i, l in enumerate(labels)}
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels, pad=15, thickness=15,
                          color=_PALETTE[:len(labels)] * (len(labels) // len(_PALETTE) + 1)),
                link=dict(source=[label_idx[s] for s in flow["source"]],
                          target=[label_idx[t] for t in flow["target"]],
                          value=flow["value"].tolist())
            )])
            fig.update_layout(title_text=title or f"Sankey: {source_col} → {target_col}",
                              template="plotly_dark")
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_sunburst(
        self,
        data: pd.DataFrame,
        path: List[str],
        values: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, str]:
        """Hierarchical sunburst chart."""
        try:
            import plotly.express as px
            fig = px.sunburst(data, path=path, values=values,
                              title=title or f"Sunburst: {' > '.join(path)}",
                              template="plotly_dark",
                              color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_funnel(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        color: Optional[str] = None,
    ) -> Dict[str, str]:
        """Funnel chart for sequential stage data."""
        try:
            import plotly.express as px
            fig = px.funnel(data, x=x, y=y, color=color,
                            title=title or f"Funnel: {y}",
                            template="plotly_dark",
                            color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_treemap(
        self,
        data: pd.DataFrame,
        path: List[str],
        values: Optional[str] = None,
        color: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, str]:
        """Treemap for hierarchical data."""
        try:
            import plotly.express as px
            fig = px.treemap(data, path=[px.Constant("All")] + path,
                             values=values, color=color,
                             title=title or f"Treemap: {' > '.join(path)}",
                             template="plotly_dark",
                             color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}

    def plot_bubble(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        size: str,
        color: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, str]:
        """Bubble chart (scatter with a third size dimension)."""
        try:
            import plotly.express as px
            fig = px.scatter(data, x=x, y=y, size=size, color=color,
                             title=title or f"Bubble: {x} vs {y} (size={size})",
                             template="plotly_dark",
                             color_discrete_sequence=_PALETTE)
            return self._wrap(fig)
        except Exception as e:
            return {"status": "error", "html": "", "message": str(e)}
