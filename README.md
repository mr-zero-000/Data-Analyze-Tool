# 📊 Data Analysis Tool

A robust Python toolkit for **data cleaning, exploratory data analysis (EDA), feature engineering, and interactive visualization**, designed specifically for **Google Colab** workflows.

The toolkit automates common preprocessing tasks such as:

* Data sanitization
* Missing value handling
* Duplicate removal
* Outlier detection
* Feature scaling & encoding
* Statistical association analysis
* Interactive visualizations

---

## ✨ Features

### 📥 Intelligent Data Loading

* Automatically recognizes common null values (`?`, `N/A`, `NULL`, etc.)
* Attempts automatic conversion of columns to appropriate numeric types
* Seamless integration with Google Colab uploads

### 🔍 Comprehensive Data Inspection

* Dataset dimensions and structure overview
* Numerical and categorical summaries
* Column datatype analysis
* Missing value reporting

### 🧹 Automated Data Cleaning

* Missing value imputation

  * Mean
  * Median
  * Mode
  * Constant value
* Duplicate row removal
* Outlier detection and removal using IQR
* Row and column deletion by index or name
* Fully Colab-compatible (no blocking prompts)

### ⚙️ Feature Engineering

#### Numeric Scaling

* Min-Max Scaling
* Standard Scaling (Z-Score)
* Robust Scaling

#### Categorical Encoding

* One-Hot Encoding
* Ordinal Encoding
* Uniform Encoding

### 📈 Interactive Visualizations

Powered by **Plotly**:

* Violin plots
* Scatter plots
* Histograms
* Box plots
* Grouped bar charts
* Pie charts
* Heatmaps
* Sankey diagrams

### 📊 Advanced Statistical Insights

#### Numerical Associations

* Pearson Correlation Heatmaps

#### Categorical Associations

* Cramér's V Heatmaps

#### Mixed-Type Associations

Unified association mapping using:

* Pearson Correlation
* Point-Biserial Correlation
* Eta (ANOVA) Association Measures

---

# 🚀 Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/mr-zero-000/Data-Analyze-Tool.git"
```

> **Note**
>
> All required plotting and machine learning dependencies (Plotly, Scikit-Learn, SciPy, etc.) are installed automatically.

---

# ⚡ Quick Start

```python
from data_analysis import DataInspector

inspector = DataInspector()
```

---

## 1️⃣ Data Cleaning & Missing Value Handling

```python
from data_analysis import DataInspector

inspector = DataInspector()

# Upload CSV from Google Colab
inspector.upload_data()

# Impute missing values
inspector.handle_missing_values(strategy='median')

# Remove duplicate rows
inspector.remove_duplicates()
```

---

## 2️⃣ Load Data from URLs or Repositories

### Public CSV

```python
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

inspector.df = pd.read_csv(url)
```

### UCI Machine Learning Repository

```python
!pip install -q ucimlrepo

from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=597)

inspector.df = dataset.data.features
```

---

## 3️⃣ Exploratory Data Analysis

### Numerical Distributions

```python
inspector.plot_numerical(['Age', 'Salary'])
```

### Variable Relationships

```python
inspector.plot_relationship('Department', 'Salary')
```

The tool automatically selects the most appropriate visualization based on column types.

---

## 4️⃣ Delete Rows and Columns

### Delete Columns

```python
# View columns
inspector.delete_columns()

# Delete selected columns
inspector.delete_columns(['PassengerId', 'Ticket'])
```

### Delete Rows

```python
# View available indices
inspector.delete_rows()

# Delete specific rows
inspector.delete_rows([0, 5, 12])
```

---

## 5️⃣ Feature Engineering & Normalization

### Numeric Scaling

```python
normalized_numeric = (
    inspector.extract_normalized_numeric_data(
        method='robust'
    )
)
```

### Categorical Encoding

```python
encoded_cat = (
    inspector.extract_normalized_categorical_data(
        method='onehot'
    )
)
```

### Combined ML-Ready Dataset

```python
final_df = inspector.create_normalized_data_df()
```

---

## 6️⃣ Statistical Association Analysis

### Numerical Correlation

```python
inspector.plot_numerical_correlation()
```

### Categorical Correlation

```python
inspector.plot_categorical_correlation()
```

### Unified Association Heatmap

```python
inspector.plot_all_associations_heatmap()
```

---

## 7️⃣ Export Cleaned Data

```python
inspector.export_cleaned_data(
    filename='cleaned_data.csv'
)
```

Automatically downloads the cleaned dataset within Google Colab.

---

# 🎨 Custom Visualization API

The `PlottingMethods` class provides direct access to individual chart generators.

```python
PLT = PlottingMethods()
```

All methods return a result dictionary that should be rendered using:

```python
PLT.display_image(result)
```

---

## Bar Charts

```python
result = PLT.plot_bar_chart(
    x='Department',
    y='Salary',
    color='Gender',
    barmode='group',
    data=inspector.df
)

PLT.display_image(result)
```

---

## Pie Charts

```python
result = PLT.plot_pie_chart(
    names='Category',
    values='Total',
    hole=0.4,
    title='Revenue Split',
    data=inspector.df
)

PLT.display_image(result)
```

---

## Histograms

```python
result = PLT.plot_histogram(
    x='Age',
    bins=[0, 18, 35, 60, 100],
    title='Age Demographics',
    data=inspector.df
)

PLT.display_image(result)
```

---

## Heatmaps

```python
result = PLT.plot_heat_map(
    values='actual_productivity',
    index='department',
    columns='quarter',
    aggregade_method='mean',
    title='Productivity Heatmap',
    data=inspector.df
)

PLT.display_image(result)
```

---

## Sankey Diagrams

```python
result = PLT.plot_sankey_diagram(
    source_column='department',
    target_column='quarter',
    values='actual_productivity',
    data=inspector.df
)

PLT.display_image(result)
```

---

# 📂 Project Structure

```text
Data-Analyze-Tool/
│
├── data_analysis/
│   ├── __init__.py
│   └── core.py
│
├── pyproject.toml
└── README.md
```

---

# 👨‍💻 Author

* **E/23/034**

---

# 📄 License

Licensed under the **MIT License**.

Feel free to use, modify, and distribute this project under the terms of the license.
ct is licensed under the MIT License.
