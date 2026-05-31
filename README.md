data-analysis-tool
A robust Python toolkit designed for data cleaning, exploration, and interactive visualization within Google Colab environments. This tool automates common preprocessing tasks such as data sanitization, missing value imputation, outlier detection, and advanced statistical association mapping.

Features

Intelligent Data Loading: Automatically handles common null strings (e.g., '?', 'N/A', 'NULL') and attempts auto-conversion of columns to correct numeric types.
Comprehensive Inspection: Quickly view data dimensions, column type breakdowns, and statistical summaries for both numerical and categorical data.
Automated Cleaning:

Identify and impute missing values using mean, median, mode, or constant strategies.
Remove exact duplicate rows and specific outliers using IQR logic.
Delete specific rows and columns by index/name — no blocking prompts, fully compatible with Colab.


Advanced Scaling & Encoding:

Numeric: Min-Max, Standard (Z-score), and Robust scaling.
Categorical: One-Hot, Ordinal, and Uniform encoding.


Interactive Visualizations: Powered by Plotly, including horizontal violin plots, scatter plots, histograms, and grouped bar charts.
Deep Statistical Insights:

Pearson Correlation heatmaps for numeric data.
Cramér's V heatmaps for categorical associations.
Unified Association Heatmaps combining Numeric and Categorical data (using Point-Biserial and Eta/ANOVA).




Installation
bash# Install directly from GitHub
pip install "git+https://github.com/mr-zero-000/Data-Analyze-Tool.git"

Note: All plotting dependencies (Plotly, scikit-learn, scipy) are included automatically.


Quick Start (Use Cases)
The tool is optimised for use in Google Colab.
1. Data Cleaning and Imputation
Load a dataset and handle missing values in one flow.
pythonfrom data_analysis import DataInspector

inspector = DataInspector()

# Step 1: Upload your CSV (interactive file-picker in Colab)
inspector.upload_data()

# Step 2: Impute missing values using the median strategy
inspector.handle_missing_values(strategy='median')

# Step 3: Remove duplicate rows
inspector.remove_duplicates()
2. Loading Data from a URL or Repository
pythonimport pandas as pd

# Load from a public URL (e.g. Titanic dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
inspector.df = pd.read_csv(url)

# Or load from the UCI ML Repository
!pip install -q ucimlrepo
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=597)
inspector.df = dataset.data.features
3. Exploratory Data Analysis (Visual)
Generate multi-chart statistical views of your variables.
python# Visualize numerical distributions (Violin, Scatter, and Histogram in one view)
inspector.plot_numerical(['Age', 'Salary'])

# Explore relationships between two variables
# (Auto-selects Scatter, Box, or Grouped Bar based on column types)
inspector.plot_relationship('Department', 'Salary')
4. Deleting Rows and Columns
python# See available columns first, then pass names to delete
inspector.delete_columns()                              # prints column list
inspector.delete_columns(['PassengerId', 'Ticket'])     # deletes by name

# See current index range first, then pass indices to delete
inspector.delete_rows()                 # prints index range
inspector.delete_rows([0, 5, 12])       # deletes rows by index
5. Feature Engineering & Normalisation
Prepare your data for Machine Learning models.
python# Scale numeric columns using Robust Scaling (better for outliers)
normalized_numeric = inspector.extract_normalized_numeric_data(method='robust')

# Encode categorical columns using One-Hot encoding
encoded_cat = inspector.extract_normalized_categorical_data(method='onehot')

# Create a single merged DataFrame ready for training
final_df = inspector.create_normalized_data_df()
6. Advanced Correlation Mapping
Identify hidden associations across different data types.
python# Pearson correlation (numeric)
inspector.plot_numerical_correlation()

# Cramér's V (categorical)
inspector.plot_categorical_correlation()

# Unified heatmap across all column types
inspector.plot_all_associations_heatmap()
7. Exporting Cleaned Data
python# Saves CSV and triggers a browser download in Colab automatically
inspector.export_cleaned_data(filename='cleaned_data.csv')
8. Custom Chart Generation
The PlottingMethods class provides direct access to specific chart types. Methods return a result dictionary; render it with PLT.display_image(result).
Bar Charts
pythonPLT = PlottingMethods()

result = PLT.plot_bar_chart(
    x='Department',
    y='Salary',
    color='Gender',
    barmode='group',
    data=inspector.df
)
PLT.display_image(result)
Pie Charts
pythonresult = PLT.plot_pie_chart(
    names='Category',
    values='Total',
    hole=0.4,
    title='Revenue Split',
    data=inspector.df
)
PLT.display_image(result)
Histograms
pythonresult = PLT.plot_histogram(
    x='Age',
    bins=[0, 18, 35, 60, 100],
    title='Age Demographics',
    data=inspector.df
)
PLT.display_image(result)
Heatmaps & Sankey Diagrams
pythonresult = PLT.plot_heat_map(
    values='actual_productivity',
    index='department',
    columns='quarter',
    aggregade_method='mean',
    title='Productivity Heatmap',
    data=inspector.df
)
PLT.display_image(result)

result = PLT.plot_sankey_diagram(
    source_column='department',
    target_column='quarter',
    values='actual_productivity',
    data=inspector.df
)
PLT.display_image(result)

Implementation Note for Google Colab: Always use PLT.display_image(result) to render charts from PlottingMethods inside a Colab notebook.


Project Structure
textData-Analyze-Tool/
├── data_analysis/
│   ├── __init__.py
│   └── core.py        # Contains DataInspector and PlottingMethods
├── pyproject.toml     # Project configuration
└── README.md

Authors

E/23.034 — GitHub


License
This project is licensed under the MIT License.
