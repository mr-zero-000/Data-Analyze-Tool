# DataWranglerTool 🛠️

A robust Python toolkit for **data wrangling**, feature engineering, interactive visualisation,
and statistical association analysis — optimised for **Google Colab**.

---

## Features

### 🔃 Intelligent Data Loading
- Uploads from local disk (Colab), URLs, or direct DataFrame assignment
- Supports CSV, TSV, XLSX/XLS, JSON, and Parquet
- Automatically neutralises common null strings (`?`, `N/A`, `NULL`, …)
- Smart column-type inference on load

### 🔍 Comprehensive Profiling
- `get_profile()` — shape, dtype breakdown, memory, duplicates, missing-value table
- `show_schema()` — column-by-column dtype, null counts, unique counts, sample values
- `column_report(col)` — deep per-column stats: skewness, kurtosis, outlier count, value-count bar chart
- `get_numeric_summary()` — extended descriptive stats including skewness & kurtosis
- `get_categorical_summary()` — value-counts + Shannon entropy per categorical column
- `show_missing_heatmap()` — visual missingness map

### 🧹 Automated Cleaning
- `fix_nulls(strategy)` — mean / median / mode / constant / ffill / bfill / drop
- `drop_duplicates()` — exact duplicate removal with configurable `keep`
- `fix_outliers(strategy)` — remove, clip, or winsorise via IQR logic
- `flag_outliers()` — returns a boolean DataFrame (non-destructive)

### ✂️ Reshaping
- `drop_columns()` — interactive (numbered list + prompt) or by name
- `drop_rows()` — interactive or by index list
- `rename_columns()` — dict mapping or interactive prompt
- `cast_columns()` — safe dtype casting including datetime parsing
- `reorder_columns()` — move columns to front, rest appended

### ⚙️ Feature Engineering
- `bin_column()` — equal-width or custom-edge binning with optional labels
- `extract_datetime_parts()` — expands year / month / day / hour / dayofweek / quarter / week
- `add_ratio_column()` — numerator ÷ denominator with safe zero-division handling
- `apply_transform()` — arbitrary function applied element-wise (`np.log1p`, `str.strip`, …)

### 🔢 Encoding & Scaling
- `encode_categoricals()` — one-hot / ordinal / label / **frequency** encoding
- `scale_numerics()` — min-max / standard (z-score) / robust / **log1p** / **sqrt** scaling
- `get_encoded_df()` — returns a fully processed ML-ready copy without modifying `self.df`

### 📊 Interactive Visualisations (Plotly, dark theme)
- `plot_numeric()` — violin + box + histogram panel per numeric column
- `plot_categorical()` — horizontal bar chart of top-N value frequencies
- `plot_pair()` — scatter matrix (pair plot) with optional colour grouping
- `plot_relationship()` — auto-selects scatter / box / grouped bar based on column types
- `plot_timeseries()` — multi-series line chart with optional resampling

### 🔗 Statistical Insights
- `plot_numeric_heatmap()` — Pearson / Spearman / Kendall correlation heatmap
- `plot_categorical_heatmap()` — Cramér's V association heatmap
- `plot_unified_heatmap()` — unified |Pearson r| + Cramér's V + η² (eta-squared) heatmap

### 💾 Export & Snapshots
- `export_csv()` / `export_excel()` — save current state to disk
- `snapshot(name)` / `restore(name)` — named rollback checkpoints
- `get_history()` — human-readable operation log

---

## Installation

```bash
# Core (no plotting)
pip install "git+https://github.com/your-org/data-wrangling-tool.git"

# With plotting support (recommended)
pip install "data-wrangling-tool[plotting]"

# Everything (including Parquet)
pip install "data-wrangling-tool[full]"
```

---

## Quick Start

```python
from data_wrangling import DataWrangler, ChartBuilder

# ── 1. Load ────────────────────────────────────────────────────────────────
wrangler = DataWrangler()

# Option A: interactive upload (Colab)
wrangler.upload_data()

# Option B: URL
wrangler.load_url("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Option C: assign directly
import pandas as pd
wrangler.df = pd.read_csv("my_data.csv")


# ── 2. Profile ─────────────────────────────────────────────────────────────
wrangler.get_profile()
wrangler.show_schema()
wrangler.column_report("Age")


# ── 3. Clean ───────────────────────────────────────────────────────────────
wrangler.fix_nulls(strategy="median")
wrangler.drop_duplicates()
wrangler.fix_outliers(strategy="clip", k=1.5)


# ── 4. Reshape ─────────────────────────────────────────────────────────────
wrangler.drop_columns(["Cabin", "Ticket"])
wrangler.rename_columns({"Pclass": "passenger_class"})
wrangler.cast_columns({"passenger_class": "category"})


# ── 5. Feature Engineering ─────────────────────────────────────────────────
wrangler.bin_column("Age", bins=[0, 12, 18, 60, 120],
                    labels=["Child", "Teen", "Adult", "Senior"])
wrangler.add_ratio_column("Fare", "Age", new_column="fare_per_age")


# ── 6. Encode & Scale ──────────────────────────────────────────────────────
ml_df = wrangler.get_encoded_df(numeric_method="robust", categorical_method="onehot")


# ── 7. Visualise ───────────────────────────────────────────────────────────
wrangler.plot_numeric(["Age", "Fare"])
wrangler.plot_categorical(["Sex", "Embarked"])
wrangler.plot_pair(color="Survived")
wrangler.plot_relationship("Sex", "Survived")


# ── 8. Correlate ───────────────────────────────────────────────────────────
wrangler.plot_numeric_heatmap()
wrangler.plot_categorical_heatmap()
wrangler.plot_unified_heatmap()


# ── 9. Snapshots & Export ──────────────────────────────────────────────────
wrangler.snapshot("before_encoding")
wrangler.export_csv("titanic_clean.csv")
wrangler.get_history()
```

---

## Custom Charts with ChartBuilder

```python
cb = ChartBuilder()

# View all available chart methods
import pandas as pd
pd.DataFrame(cb.get_methods_info()["response"])

# Bar chart
result = cb.plot_bar(wrangler.df, x="Sex", y="Fare", color="Survived", barmode="group")
cb.display(result)

# Donut pie
result = cb.plot_pie(wrangler.df, names="Embarked", values="Fare", hole=0.4)
cb.display(result)

# Histogram with KDE rug
result = cb.plot_histogram(wrangler.df, x="Age", kde=True)
cb.display(result)

# Sankey flow
result = cb.plot_sankey(wrangler.df, source_col="Sex", target_col="Survived", value_col="Fare")
cb.display(result)

# Treemap
result = cb.plot_treemap(wrangler.df, path=["Sex", "Pclass"], values="Fare")
cb.display(result)

# Bubble
result = cb.plot_bubble(wrangler.df, x="Age", y="Fare", size="SibSp", color="Survived")
cb.display(result)
```

---

## Project Structure

```
data-wrangling-tool/
├── data_wrangling/
│   ├── __init__.py          # Public API: DataWrangler, ChartBuilder
│   └── core.py              # Full implementation
├── notebooks/
│   └── quickstart.ipynb     # Colab-ready walkthrough notebook
├── pyproject.toml
└── README.md
```

---

## Comparison with Similar Tools

| Feature                        | DataWranglerTool | pandas-profiling | sweetviz |
|-------------------------------|:----------------:|:----------------:|:--------:|
| Interactive Colab upload       | ✅               | ❌               | ❌       |
| Snapshot / rollback            | ✅               | ❌               | ❌       |
| Frequency encoding             | ✅               | ❌               | ❌       |
| log1p / sqrt scaling           | ✅               | ❌               | ❌       |
| Unified assoc. heatmap         | ✅               | ⚠️ partial      | ❌       |
| Funnel / treemap / bubble      | ✅               | ❌               | ❌       |
| Operation history log          | ✅               | ❌               | ❌       |

---

## License

MIT License.
