from __future__ import annotations
from importlib.resources import files
from typing import Optional, Sequence, Tuple, Dict, Any, List
import os
import io
import json
import uuid
import pandas as pd
import numpy as np
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from IPython.display import display, HTML

def _is_colab():
    """Detect whether we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

class DataInspector:
    """
    A comprehensive data cleaning and exploration tool optimized for Google Colab and Jupyter environments.
    Provides interactive visualizations using Plotly and robust data sanitization.
    """

    def __init__(self):
        self.df = None
        self.numeric_df = None
        self.categorical_df = None
        self.categorical_normalized_df = None
        self.normalized_data_df = None
        self.numeric_normalized_df = None

    def upload_data(self):
        """
        Uploads a CSV file interactively.
        - In Google Colab: opens the browser file-picker via google.colab.files.upload().
        - In local Jupyter: prompts for a file path via input().
        """
        if _is_colab():
            from google.colab import files as colab_files
            print("📂 Click the button below to choose a CSV file from your computer.")
            uploaded = colab_files.upload()
            if not uploaded:
                return print("❌ No file was uploaded.")
            file_name = next(iter(uploaded))
            file_bytes = io.BytesIO(uploaded[file_name])
            print(f"\n🔄 Processing uploaded file: {file_name}")
        else:
            import os
            print("====== DATA LOADING INTERFACE ======")
            print("1. Upload a custom .csv file from your machine")
            print("2. Open the local 'titanic.csv' file")
            print("====================================")
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                file_name = input("Enter the path to your CSV file: ").strip()
                if not os.path.exists(file_name):
                    return print(f"❌ Error: File '{file_name}' does not exist.")
                file_bytes = file_name
                print(f"\n🔄 Loading file: {file_name}")
            elif choice == "2":
                file_name = "titanic.csv"
                if not os.path.exists(file_name):
                    return print(f"❌ Error: Could not find '{file_name}' in your local directory path.")
                file_bytes = file_name
                print(f"\n🔄 Opening local dataset: {file_name}")
            else:
                return print("❌ Invalid selection. Please enter 1 or 2.")

        self.df = pd.read_csv(file_bytes, na_values=['?', 'n/a', 'N/A', 'NULL', 'null', ' '])
        self.df['count'] = 1
        for col in self.df.columns:
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')
            if not numeric_col.isna().all():
                self.df[col] = numeric_col
        print(f"✅ File '{file_name}' successfully loaded and types sanitized!")

    def get_summary(self):
        """
        Prints data dimensions, column type breakdown, 
        and displays ALL rows of the DataFrame inline.
        """
        if self.df is None: 
            return print("Error: No data loaded.")

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in num_cols:
            num_cols.remove('count')
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print(f"--- Data Summary ---")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")
        print(f"Numerical ({len(num_cols)}): {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")
        
        # This temporarily forces Jupyter/pandas to show ALL rows and columns
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(self.df)

    def show_missing_data(self, column=None):
        """
        Shows rows with missing values.
        - If *column* is provided, also prints a positional missing-value mask for that column.
        - Pass column='ColumnName' to inspect a specific column.
        Compatible with Google Colab and local Jupyter (no blocking input() calls).
        """
        if self.df is None: 
            return print("Error: No data loaded.")
            
        missing_mask = self.df.isnull().any(axis=1) | (self.df == "").any(axis=1)
        missing_rows = self.df[missing_mask]

        if missing_rows.empty:
            print("✨ No missing data found!")
            return
            
        print(f"🔍 Found {len(missing_rows)} rows with missing values globally.")
        
        if column is not None:
            if column not in self.df.columns:
                print(f"❌ Error: Column '{column}' does not exist in dataset.")
            else:
                pos_series = self.df[column].isnull() | (self.df[column] == "")
                pos_display = pos_series.map({True: 'yes', False: 'no'}).to_frame(name='missing_values')
                print(f"\n--- Positional Missing Mask for column: '{column}' ---")
                with pd.option_context('display.max_rows', None):
                    print(pos_display)
        else:
            print("💡 Tip: call show_missing_data(column='ColumnName') to inspect a specific column.")
            display(missing_rows.head(20))

    def delete_rows(self, indices=None):
        """
        Deletes rows by index.
        Args:
            indices: a list of integer row indices to drop, e.g. [1, 3, 15].
                     If None, prints instructions and current index range.
        Compatible with Google Colab and local Jupyter (no blocking input() calls).
        """
        if self.df is None: return
        if indices is None:
            print(f"💡 Pass a list of row indices to delete, e.g.: inspector.delete_rows([1, 3, 15])")
            print(f"   Current index range: 0 – {len(self.df) - 1}")
            return
        try:
            if isinstance(indices, str):
                indices = [int(i.strip()) for i in indices.split(',') if i.strip().isdigit()]
            existing_indices = [i for i in indices if i in self.df.index]
            self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
            print(f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def delete_columns(self, columns=None):
        """
        Deletes columns by name.
        Args:
            columns: a list of column names to drop, e.g. ['Age', 'Cabin'].
                     If None, prints the current column list.
        Compatible with Google Colab and local Jupyter (no blocking input() calls).
        """
        if self.df is None:
            return print("No data loaded.")

        if columns is None:
            print(f"💡 Pass a list of column names to delete, e.g.: inspector.delete_columns(['Age', 'Cabin'])")
            print(f"   Current columns: {', '.join(self.df.columns)}")
            return

        try:
            if isinstance(columns, str):
                columns = [c.strip() for c in columns.split(',')]
            existing_cols = [c for c in columns if c in self.df.columns]

            if not existing_cols:
                return print("⚠️ None of the provided column names were found.")

            self.df = self.df.drop(columns=existing_cols)
            print(f"🗑️ Deleted {len(existing_cols)} columns. Remaining count: {len(self.df.columns)}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def handle_missing_values(self, columns=None, strategy='median', fill_value=None):
        """
        Imputes missing values in specified columns.
        """
        if self.df is None: return
        target_cols = columns if columns else self.df.columns[self.df.isnull().any()].tolist()

        for col in target_cols:
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == 'mode':
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)

        print(f"🛠️ Imputation complete using '{strategy}' strategy for: {target_cols}")

    def remove_duplicates(self):
        """
        Identifies and removes exact duplicate rows from the DataFrame.
        """
        if self.df is None: return
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(self.df)
        print(f"✨ Removed {dropped} duplicate rows. New row count: {len(self.df)}")

    def export_cleaned_data(self, filename='cleaned_data.csv'):
        """
        Exports the current dataset as a CSV file.
        - In Google Colab: saves to /content/<filename> and triggers a browser download.
        - In local Jupyter: saves to the current working directory.
        """
        if self.df is None: return
        self.df.to_csv(filename, index=False)
        print(f"💾 Cleaned data saved as: '{filename}'")
        if _is_colab():
            from google.colab import files as colab_files
            colab_files.download(filename)
            print("⬇️  Download triggered in your browser.")
        else:
            print(f"📁 Full path: '{os.path.abspath(filename)}'")

    def column_details(self):
        """
        Iterates through all columns to show numeric ranges or categorical unique value counts.
        """
        if self.df is None: return
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"🔹 {col} (Numeric): Range [{self.df[col].min()} to {self.df[col].max()}]")
            else:
                print(f"🔸 {col} (Categorical): {self.df[col].nunique()} unique values")

    def get_categorical_summary(self):
        """
        Generates statistical summaries for categorical columns.
        """
        if self.df is None: return
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("No categorical columns found.")

        summary = cat_df.describe().T[['unique', 'top', 'freq']]
        print("--- Categorical Deep Dive ---")
        display(summary)

    def extract_numeric_data(self):
        if self.df is None: return print("Error: No data loaded.")
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df

    def extract_categorical_data(self):
        if self.df is None: return print("Error: No data loaded.")
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df

    def extract_normalized_numeric_data(self, method='minmax'):
        if self.df is None: return print("Error: No data loaded.")
        num_df = self.df.select_dtypes(include=[np.number]).copy()

        if num_df.empty:
            print("⚠️ No numerical columns found to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df

        if num_df.isnull().any().any():
            print("ℹ️ Missing values detected. Imputing with column medians before scaling...")
            num_df = num_df.fillna(num_df.median())

        method_lower = method.lower().strip()
        if method_lower == 'minmax':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)
        elif method_lower == 'standard':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)
        elif method_lower == 'robust':
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(num_df)
            self.numeric_normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns, index=num_df.index)
        else:
            print(f"❌ Unknown scaling method '{method}'. Defaulting to 'minmax'.")
            return self.extract_normalized_numeric_data(method='minmax')

        print(f"✨ Successfully scaled numerical data using the '{method_lower}' method.")
        return self.numeric_normalized_df
        
    def sample_data(self, n=20):
        """
        Randomly selects a specified number of samples (default 20) from the 
        population DataFrame for downstream plotting workflows.
        """
        if self.df is None:
            return print("❌ Error: No data loaded to sample from.")
        
        if len(self.df) <= n:
            self.plotting_df = self.df.copy()
            print(f"ℹ️ Dataset has {len(self.df)} rows, which is less than or equal to {n}. Using all rows for plotting.")
        else:
            self.plotting_df = self.df.sample(n=n, random_state=42).reset_index(drop=True)
            print(f"🎲 Randomly sampled {n} rows from a population of {len(self.df)} rows for plotting works.")
        
        return self.plotting_df

    def select_columns_for_plotting(self, columns=None) -> list:
        """
        Returns the list of columns to use for plotting.
        Args:
            columns: list of column names/numbers, a comma-separated string, or 'all'.
                     If None, prints available columns and returns an empty list.
        Compatible with Google Colab and local Jupyter (no blocking input() calls).
        """
        if self.df is None:
            print("❌ Error: No data loaded.")
            return []
            
        available_cols = list(self.df.columns)
        if 'count' in available_cols:
            available_cols.remove('count')
            
        if columns is None:
            print("\n--- Available Database Columns ---")
            for i, col in enumerate(available_cols, 1):
                print(f"{i}. {col}")
            print("---------------------------------")
            print("💡 Pass columns=['col1','col2'] or columns='all' to this method, or supply column_names directly to plot_numerical().")
            return []

        if isinstance(columns, list):
            return [c for c in columns if c in available_cols]

        user_input = str(columns).strip()
        selected_cols = []
        if user_input.lower() == 'all':
            selected_cols = available_cols
        else:
            items = [item.strip() for item in user_input.split(',')]
            for item in items:
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(available_cols):
                        selected_cols.append(available_cols[idx])
                elif item in available_cols:
                    selected_cols.append(item)
                    
        print(f"🎯 Selected for plotting: {selected_cols}")
        return selected_cols
    def plot_numerical(self, column_names: list):
        """
        Accepts any completely variable list of columns. Plots exact sets of 
        multi-grid structural distributions dynamically based on whatever data is sampled.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        # Fallback to the full dataframe if sample_data() wasn't executed beforehand
        source_df = getattr(self, 'plotting_df', self.df)
        if source_df is None:
            source_df = self.df

        # Filter out invalid or non-numeric columns from the user input safely
        valid_cols = [c for c in column_names if c in source_df.columns and pd.api.types.is_numeric_dtype(source_df[c])]

        if not valid_cols:
            return print("⚠️ No valid numeric columns found in the selection matching this database schema.")

        print(f"📊 Generating continuous multi-chart views for {len(valid_cols)} columns using {len(source_df)} sample rows...")

        for col in valid_cols:
            if col == 'count': 
                continue
                
            # Exact layout tracking requirements (Violin/Box, Scatter distribution, Histogram)
            fig = make_subplots(
                rows=1, cols=3, 
                subplot_titles=(
                    f"Horizontal Violin/Box: {col}", 
                    f"Scatter Distribution: {col}", 
                    f"Histogram: {col}"
                )
            )
            
            # 1. Horizontal Violin/Box Layout
            fig.add_trace(
                go.Violin(
                    x=source_df[col], 
                    box_visible=True, 
                    meanline_visible=True, 
                    name=col, 
                    orientation='h', 
                    line_color='lightseagreen'
                ), 
                row=1, col=1
            )
            
            # 2. Sequential Data Points Distribution
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(source_df))),
                    y=source_df[col], 
                    mode='markers+lines', 
                    marker=dict(opacity=0.7, color='royalblue', size=6),
                    name=col
                ), 
                row=1, col=2
            )
            
            # 3. Frequency Binned Graph
            fig.add_trace(
                go.Histogram(
                    x=source_df[col], 
                    name=col, 
                    marker_color='indianred'
                ), 
                row=1, col=3
            )
            
            fig.update_layout(
                height=400, 
                width=1100,
                title_text=f"<b>Variable Statistical Matrix Profile: [ {col} ]</b>", 
                showlegend=False, 
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Value Range", row=1, col=1)
            fig.update_xaxes(title_text="Sample Index Row", row=1, col=2)
            fig.update_yaxes(title_text="Value Range", row=1, col=2)
            fig.update_xaxes(title_text="Value Range", row=1, col=3)
            fig.update_yaxes(title_text="Frequency Count", row=1, col=3)
            
            fig.show()

   # =========================================================================
    # 1. COLUMN DETAILS
    # =========================================================================
    def column_details(self):
        """
        Iterates through all columns to show numeric ranges or categorical unique value counts.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        print("\n--- Detailed Column Breakdown ---")
        for col in self.df.columns:
            if col == 'count':
                continue
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                print(f"🔹 {col:<25} (Numeric)     -> Range: [{col_min} to {col_max}]")
            # Otherwise, handle as categorical
            else:
                unique_count = self.df[col].nunique()
                print(f"🔸 {col:<25} (Categorical) -> Unique Values: {unique_count}")

    # =========================================================================
    # 2. CATEGORICAL SUMMARY
    # =========================================================================
    def get_categorical_summary(self):
        """
        Generates a detailed statistical summary for categorical columns, 
        including unique counts, the most frequent value (Mode), and its frequency.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("⚠️ No categorical columns found in this database.")
            
        print("\n--- Categorical Statistical Summary ---")
        # Generate description metrics
        summary = cat_df.describe().T[['unique', 'top', 'freq']]
        # Rename columns for clarity matching requirements
        summary.columns = ['Unique Counts', 'Most Frequent Value (Mode)', 'Frequency']
        display(summary)

    # =========================================================================
    # 3. EXTRACT & LOCAL EXPORT (DOWNLOAD) FOR NUMERIC/CATEGORICAL DATA
    # =========================================================================
    def extract_numeric_data(self, filename="extracted_numeric_data.csv"):
        """
        Filters the DataFrame to include only numeric columns and saves it as a local CSV file.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        self.numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'count' in self.numeric_df.columns:
            self.numeric_df = self.numeric_df.drop(columns=['count'])
            
        if self.numeric_df.empty:
            return print("⚠️ No numerical columns available to extract.")
            
        file_path = os.path.abspath(filename)
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        open(file_path, "w", encoding="utf-8", newline="").close()
        self.numeric_df.to_csv(file_path, index=False)
        print(f"💾 Numeric columns filtered and saved locally to: '{file_path}'")
        return self.numeric_df

    def extract_categorical_data(self, filename="extracted_categorical_data.csv"):
        """
        Filters the DataFrame to include only categorical (non-numeric) columns and saves it as a local CSV file.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        self.categorical_df = self.df.select_dtypes(exclude=[np.number]).copy()
        
        if self.categorical_df.empty:
            return print("⚠️ No categorical columns available to extract.")
            
        file_path = os.path.abspath(filename)
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        open(file_path, "w", encoding="utf-8", newline="").close()
        self.categorical_df.to_csv(file_path, index=False)
        print(f"💾 Categorical columns filtered and saved locally to: '{file_path}'")
        return self.categorical_df

    # =========================================================================
    # 4. EXTRACT NORMALIZED NUMERIC DATA
    # =========================================================================
    def extract_normalized_numeric_data(self, method='minmax'):
        """
        Extracts numerical columns and scales them using the specified method.
        
        Parameters:
        - method: str
            * 'minmax': Scales features exactly to the [0, 1] range.
            * 'standard': Centers features to a mean of 0 and standard deviation of 1.
            * 'robust': Uses the median and Interquartile Range (IQR). Best for outliers.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'count' in num_df.columns:
            num_df = num_df.drop(columns=['count'])
            
        if num_df.empty:
            print("⚠️ No numerical columns found to normalize.")
            return pd.DataFrame()

        # Handle any outstanding missing values prior to mathematical scaling computations
        if num_df.isnull().any().any():
            num_df = num_df.fillna(num_df.median())

        method_lower = method.lower().strip()
        if method_lower == 'minmax':
            scaler = MinMaxScaler()
        elif method_lower == 'standard':
            scaler = StandardScaler()
        elif method_lower == 'robust':
            scaler = RobustScaler()
        else:
            print(f"❌ Unknown scaling method '{method}'. Defaulting to 'minmax'.")
            scaler = MinMaxScaler()
            method_lower = 'minmax'

        scaled_array = scaler.fit_transform(num_df)
        self.numeric_normalized_df = pd.DataFrame(scaled_array, columns=num_df.columns, index=num_df.index)
        
        print(f"✨ Successfully scaled numerical data using '{method_lower}' method.")
        return self.numeric_normalized_df

    # =========================================================================
    # 5. EXTRACT NORMALIZED CATEGORICAL DATA
    # =========================================================================
    def extract_normalized_categorical_data(self, method='uniform'):
        """
        Extracts categorical columns and applies the specified encoding method.
        
        Parameters:
        - method: str
            * 'uniform': Maps categories to numeric codes scaled to the [0, 1] range.
            * 'ordinal': Converts categories to distinct integers (0, 1, 2...) using OrdinalEncoder.
            * 'onehot': Converts categories to multiple binary (0 or 1) columns using OneHotEncoder.
            * 'minmax_ordinal': First encodes ordinally, then scales to exactly [0, 1] using MinMaxScaler.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")
            
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty:
            print("⚠️ No categorical columns found to encode.")
            return pd.DataFrame()

        # Impute missing strings to avoid encoder failures
        cat_df_filled = cat_df.fillna("Missing")
        method_lower = method.lower().strip()

        if method_lower == 'uniform':
            self.categorical_normalized_df = pd.DataFrame(index=cat_df.index)
            for col in cat_df.columns:
                codes = cat_df[col].astype('category').cat.codes
                max_code = codes.max()
                self.categorical_normalized_df[col] = codes / max_code if max_code > 0 else 0.0
                
        elif method_lower == 'ordinal':
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(cat_df_filled)
            self.categorical_normalized_df = pd.DataFrame(encoded, columns=cat_df.columns, index=cat_df.index)
            
        elif method_lower == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(cat_df_filled)
            feature_names = encoder.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded, columns=feature_names, index=cat_df.index)
            
        elif method_lower == 'minmax_ordinal':
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            encoded_ordinal = encoder.fit_transform(cat_df_filled)
            scaled_minmax = scaler.fit_transform(encoded_ordinal)
            self.categorical_normalized_df = pd.DataFrame(scaled_minmax, columns=cat_df.columns, index=cat_df.index)
            
        else:
            print(f"❌ Unknown encoding method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method='uniform')

        print(f"✨ Successfully encoded categorical data using '{method_lower}' method.")
        return self.categorical_normalized_df
    
    # =========================================================================
    # 1. MERGE NORMALIZED WORKSPACE DATAFRAME
    # =========================================================================
    def create_normalized_data_df(self) -> pd.DataFrame:
        """
        Creates a single DataFrame containing the original numeric columns 
        merged side-by-side with the normalized categorical columns.
        """
        if self.df is None:
            return print("❌ Error: No data loaded.")

        # Gather original numeric columns (excluding our tracking helper column)
        num_orig_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'count' in num_orig_df.columns:
            num_orig_df = num_orig_df.drop(columns=['count'])

        # Fallback extraction to default 'uniform' if categorical normalization hasn't been run yet
        cat_norm_df = getattr(self, 'categorical_normalized_df', None)
        if cat_norm_df is None:
            print("ℹ️ Categorical normalization wasn't executed previously. Running default 'uniform' encoding...")
            cat_norm_df = self.extract_normalized_categorical_data(method='uniform')

        # Handle empty situations gracefully depending on what columns exist in the uploaded database
        if cat_norm_df.empty:
            self.normalized_data_df = num_orig_df
        elif num_orig_df.empty:
            self.normalized_data_df = cat_norm_df
        else:
            # Merge side-by-side matching index alignment
            self.normalized_data_df = pd.concat([num_orig_df, cat_norm_df], axis=1)

        print(f"✅ Success! Created merged data profile containing {self.normalized_data_df.shape[1]} features.")
        with pd.option_context('display.max_columns', None):
            display(self.normalized_data_df.head(5))
        return self.normalized_data_df

    # =========================================================================
    # 2. PLOT NUMERICAL (UPGRADED WITH HORIZONTAL SWAPPING)
    # =========================================================================
    def plot_numerical(self, column_names: list):
        """
        Generates interactive Horizontal Violin, Scatter, and Histogram plots.
        Swapping the axis for Violin/Box plots to improve horizontal comparison.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        # Use random sampled data if available, fallback to full population
        source_df = getattr(self, 'plotting_df', self.df)
        if source_df is None:
            source_df = self.df

        valid_cols = [c for c in column_names if c in source_df.columns and pd.api.types.is_numeric_dtype(source_df[c])]
        if not valid_cols:
            return print("⚠️ No valid numeric columns found matching selection criteria.")

        for col in valid_cols:
            if col == 'count': 
                continue
                
            fig = make_subplots(
                rows=1, cols=3, 
                subplot_titles=(
                    f"Horizontal Violin/Box: {col}", 
                    f"Scatter Distribution: {col}", 
                    f"Histogram: {col}"
                )
            )
            
            # Swapped Axis -> 'x=source_df[col]' maps values horizontally with 'orientation=h'
            fig.add_trace(
                go.Violin(
                    x=source_df[col], 
                    box_visible=True, 
                    meanline_visible=True, 
                    name=col, 
                    orientation='h', 
                    line_color='lightseagreen'
                ), 
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(source_df))),
                    y=source_df[col], 
                    mode='markers+lines', 
                    marker=dict(opacity=0.7, color='royalblue', size=6),
                    name=col
                ), 
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=source_df[col], 
                    name=col, 
                    marker_color='indianred'
                ), 
                row=1, col=3
            )
            
            fig.update_layout(
                height=380, width=1150,
                title_text=f"<b>Variable Statistical Matrix Profile: [ {col} ]</b>", 
                showlegend=False, template="plotly_white"
            )
            fig.update_xaxes(title_text="Value Range", row=1, col=1)
            fig.update_xaxes(title_text="Sample Index Row", row=1, col=2)
            fig.update_yaxes(title_text="Value Range", row=1, col=2)
            fig.update_xaxes(title_text="Value Range", row=1, col=3)
            fig.update_yaxes(title_text="Frequency Count", row=1, col=3)
            fig.show()

    # =========================================================================
    # 3. PLOT CATEGORICAL 
    # =========================================================================
    def plot_categorical(self, column_names: list):
        """
        Generates interactive Bar charts for categorical columns showing counts and percentages.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        source_df = getattr(self, 'plotting_df', self.df)
        if source_df is None:
            source_df = self.df

        valid_cols = [c for c in column_names if c in source_df.columns]
        
        for col in valid_cols:
            # Calculate value counts and absolute percentages manually
            counts_series = source_df[col].value_counts()
            total_samples = counts_series.sum()
            
            df_counts = counts_series.reset_index()
            df_counts.columns = [col, 'Counts']
            df_counts['Percentage'] = ((df_counts['Counts'] / total_samples) * 100).round(1)
            
            # Format text label displayed on top of the bars
            df_counts['Text_Label'] = df_counts.apply(lambda r: f"{r['Counts']} ({r['Percentage']}%)", axis=1)
            
            fig = px.bar(
                df_counts, x=col, y='Counts', text='Text_Label',
                title=f"<b>Frequency Distribution & Proportions: [ {col} ]</b>",
                color=col, color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            fig.show()

    # =========================================================================
    # 4. ROBUST AUTOMATED OUTLIER HANDLING (IQR LOGIC)
    # =========================================================================
    def handle_outliers(self, columns=None, find_and_delete=False):
        """
        Flags outliers using IQR logic. Optionally deletes the flagged rows 
        and updates the class instance. Displays relevant steps to the user attractively.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')
            
        all_outlier_indices = set()
        
        print("\n" + "="*50)
        print(" 🚨 OUTLIER DETECTION MONITOR (IQR METHOD) ")
        print("="*50)
        
        for col in target_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Extract items breaking statistical fences
            outliers_df = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers_df.empty:
                all_outlier_indices.update(outliers_df.index.tolist())
                print(f"📍 Column '{col:<20}' -> Found {len(outliers_df):<4} outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")
                
        print("-"*50)
        print(f"📊 Total unique rows flagged across all specified columns: {len(all_outlier_indices)}")
        print("-"*50)

        if all_outlier_indices:
            print("\n📋 Flagged Outlier Records Preview:")
            display(self.df.loc[list(all_outlier_indices)].head(10))
            
            if find_and_delete:
                initial_shape = self.df.shape[0]
                self.df = self.df.drop(index=list(all_outlier_indices)).reset_index(drop=True)
                print(f"\n🗑️ Clean Action Executed: Dropped {len(all_outlier_indices)} outlier rows.")
                print(f"📉 Workspace population size updated from {initial_shape} rows down to {self.df.shape[0]} rows.")
        else:
            print("✨ Perfect! No statistical outliers detected inside selected criteria features.")
        print("="*50 + "\n")

    # =========================================================================
    # 5. DYNAMIC RELATIONSHIP ROUTER
    # =========================================================================
    def plot_relationship(self, col1: str, col2: str):
        """
        Intelligently selects the best interactive plot based on column types:
        1. Num vs Num: Scatter with Trendline
        2. Cat vs Num: Box plot with data points
        3. Cat vs Cat: Grouped bar chart
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        source_df = getattr(self, 'plotting_df', self.df)
        if source_df is None:
            source_df = self.df
            
        if col1 not in source_df.columns or col2 not in source_df.columns:
            return print("❌ Error: One or both of selected column names do not exist.")

        is_num1 = pd.api.types.is_numeric_dtype(source_df[col1]) and col1 != 'count'
        is_num2 = pd.api.types.is_numeric_dtype(source_df[col2]) and col2 != 'count'

        # Case 1: Numerical vs Numerical
        if is_num1 and is_num2:
            if source_df[col1].nunique() <= 1 or source_df[col2].nunique() <= 1:
                print(f"⚠️ Constant input detected for '{col1}' or '{col2}'. Rendering scatter without trendline.")
                fig = px.scatter(source_df, x=col1, y=col2,
                                 title=f"<b>Scatter Relationship: {col1} vs {col2}</b>")
            else:
                print(f"📈 [Num vs Num] Detected: Dispatching Scatter with Trendline for '{col1}' vs '{col2}'")
                fig = px.scatter(source_df, x=col1, y=col2, trendline="ols", 
                                 title=f"<b>Scatter Relationship & OLS Trendline: {col1} vs {col2}</b>")
        
        # Case 2: Categorical vs Numerical (or vice versa)
        elif is_num1 != is_num2:
            num_col, cat_col = (col1, col2) if is_num1 else (col2, col1)
            print(f"📦 [Cat vs Num] Detected: Dispatching Box plot with full observations for '{cat_col}' vs '{num_col}'")
            fig = px.box(source_df, x=cat_col, y=num_col, points="all", color=cat_col,
                         title=f"<b>Distribution Analysis: {num_col} stratified across {cat_col}</b>")
        
        # Case 3: Categorical vs Categorical
        else:
            print(f"📊 [Cat vs Cat] Detected: Dispatching Contingency Grouped Bar Matrix for '{col1}' vs '{col2}'")
            # Build an explicit crosstab counting combinations
            contingency_df = source_df.groupby([col1, col2]).size().reset_index(name='Observations Count')
            fig = px.bar(contingency_df, x=col1, y='Observations Count', color=col2, barmode="group",
                         title=f"<b>Cross-Categorical Composition: {col1} relative to {col2}</b>")

        fig.update_layout(template="plotly_white")
        fig.show()

    # =========================================================================
    # 6. PEARSON NUMERICAL CORRELATION HEATMAP
    # =========================================================================
    def plot_numerical_correlation(self):
        """
        Displays an interactive Heatmap of the Pearson Correlation matrix 
        for all numeric features in the dataset.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'count' in num_df.columns:
            num_df = num_df.drop(columns=['count'])
            
        if num_df.shape[1] < 2:
            return print("⚠️ Not enough numerical features available to generate a correlation matrix.")

        variable_columns = num_df.loc[:, num_df.nunique() > 1]
        if variable_columns.shape[1] < 2:
            return print("⚠️ Not enough variable numerical features available to generate a correlation matrix.")

        # ensure identity matrix matches number of numeric columns (not number of rows)
        corr_matrix = pd.DataFrame(np.eye(num_df.shape[1]), index=num_df.columns, columns=num_df.columns)
        sub_corr = variable_columns.corr(method='pearson')
        corr_matrix.loc[sub_corr.index, sub_corr.columns] = sub_corr
        
        fig = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale='RdBu_r', 
            zmin=-1.0, zmax=1.0,
            title="<b>Pearson Product-Moment Correlation Heatmap Matrix</b>"
        )
        fig.update_layout(width=700, height=600, template="plotly_white")
        fig.show()

    # =========================================================================
    # 7. CRAMÉR'S V CATEGORICAL HEATMAP
    # =========================================================================
    def plot_categorical_correlation(self):
        """
        Calculates the Cramér's V association matrix for all categorical columns 
        and displays it as an interactive Plotly Heatmap.
        """
        if self.df is None: 
            return print("❌ Error: No dataset available.")
            
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.shape[1] < 2:
            return print("⚠️ Not enough categorical variables available to compute Cramér's V associations.")
            
        cols = cat_df.columns
        n_cols = len(cols)
        cramers_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
        
        for i in range(n_cols):
            for j in range(i, n_cols):
                c1, c2 = cols[i], cols[j]
                if i == j:
                    cramers_matrix.loc[c1, c2] = 1.0
                    continue
                    
                # Build localized absolute contingency counts matrix 
                contingency_table = pd.crosstab(cat_df[c1], cat_df[c2])
                if contingency_table.size == 0 or min(contingency_table.shape) <= 1:
                    continue
                
                # Chi-squared test application
                chi2 = chi2_contingency(contingency_table)[0]
                n = contingency_table.sum().sum()
                
                # Mathematically calculate Cramér's V formula coefficient range bounded [0, 1]
                v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1))) if n > 0 else 0.0
                cramers_matrix.loc[c1, c2] = v
                cramers_matrix.loc[c2, c1] = v
                
        fig = px.imshow(
            cramers_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="Viridis", 
            zmin=0.0, zmax=1.0,
            title="<b>Cramér's V Categorical Association Strength Matrix</b>"
        )
        fig.update_layout(width=720, height=620, template="plotly_white")
        fig.show()

    # =========================================================================
    # 1. COMPUTE INDIVIDUAL NUMERIC TO CATEGORICAL ASSOCIATIONS
    # =========================================================================
    def correlate_num_to_cat(self) -> pd.DataFrame:
        """
        Computes associations between all numeric and categorical columns.
        - Uses Point-Biserial correlation for binary categories (-1 to 1).
        - Uses Eta (from ANOVA) for multi-class categories (0 to 1).
        """
        if self.df is None:
            print("❌ Error: No dataset available.")
            return pd.DataFrame()

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in num_cols:
            num_cols.remove('count')
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not num_cols or not cat_cols:
            print("⚠️ Dataset must contain both numeric and categorical columns to calculate cross-associations.")
            return pd.DataFrame()

        results = []
        for num in num_cols:
            for cat in cat_cols:
                # Clean missing tokens locally for this specific calculation pair
                valid_data = self.df[[num, cat]].dropna()
                if valid_data.empty:
                    continue

                unique_cats = valid_data[cat].unique()
                if len(unique_cats) < 2:
                    continue

                # Case 1: Point-Biserial correlation for exactly binary groupings
                if len(unique_cats) == 2:
                    try:
                        # Convert binary categories to 0 and 1 flags dynamically
                        binary_mapped = (valid_data[cat] == unique_cats[0]).astype(int)
                        if binary_mapped.nunique() < 2 or valid_data[num].nunique() < 2:
                            val, method = 0.0, "Point-Biserial (constant input)"
                        else:
                            r_pb, _ = pointbiserialr(binary_mapped, valid_data[num])
                            if np.isnan(r_pb):
                                val, method = 0.0, "Point-Biserial (undefined)"
                            else:
                                val, method = r_pb, "Point-Biserial"
                    except Exception:
                        val, method = 0.0, "Failed"
                
                # Case 2: Eta Coefficient derived from one-way ANOVA for multi-class splits
                else:
                    try:
                        groups = [valid_data[valid_data[cat] == cls][num].values for cls in unique_cats]
                        groups = [g for g in groups if len(g) > 0]
                        if len(groups) > 1:
                            # Calculate Eta-Squared directly from group sums of squares
                            grand_mean = valid_data[num].mean()
                            ss_total = ((valid_data[num] - grand_mean) ** 2).sum()
                            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                            eta = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                            val, method = eta, "Eta (ANOVA)"
                        else:
                            val, method = 0.0, "Insufficient Groups"
                    except Exception:
                        val, method = 0.0, "Failed"

                results.append({
                    "Numerical Feature": num,
                    "Categorical Feature": cat,
                    "Association Metric": round(val, 4),
                    "Statistical Method": method
                })

        res_df = pd.DataFrame(results)
        print("\n--- Cross-Type Variable Associations ---")
        display(res_df)
        return res_df

    # =========================================================================
    # 2. GLOBAL UNIFIED ASSOCIATION HEATMAP
    # =========================================================================
    def plot_all_associations_heatmap(self) -> pd.DataFrame:
        """
        Creates a unified association matrix for BOTH categorical and numeric data 
        and displays it as a single interactive Plotly Heatmap.
        """
        if self.df is None:
            print("❌ Error: No dataset available.")
            return pd.DataFrame()

        # Gather dynamic valid variables excluding tracker elements
        cols = [c for c in self.df.columns if c != 'count']
        n_cols = len(cols)
        
        # 1. Initialize empty matrix 
        assoc_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
        
        # 2. Iterate through every pair of columns
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue
                
                valid_data = self.df[[col1, col2]].dropna()
                if valid_data.empty:
                    continue
                
                is_num1 = pd.api.types.is_numeric_dtype(valid_data[col1])
                is_num2 = pd.api.types.is_numeric_dtype(valid_data[col2])
                
                # Metric Type A: Numeric vs Numeric (Pearson)
                if is_num1 and is_num2:
                    if valid_data[col1].nunique() <= 1 or valid_data[col2].nunique() <= 1:
                        val = 0.0
                    else:
                        val = abs(valid_data[col1].corr(valid_data[col2], method='pearson'))
                
                # Metric Type B: Categorical vs Categorical (Cramér's V)
                elif not is_num1 and not is_num2:
                    ct = pd.crosstab(valid_data[col1], valid_data[col2])
                    if ct.size > 0 and min(ct.shape) > 1:
                        chi2 = chi2_contingency(ct)[0]
                        n = ct.sum().sum()
                        val = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else 0.0
                    else:
                        val = 0.0
                
                # Metric Type C: Mixed Numeric and Categorical (Eta/ANOVA)
                else:
                    cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
                    categories = valid_data[cat_col].unique()
                    if len(categories) > 1:
                        groups = [valid_data[valid_data[cat_col] == c][num_col].values for c in categories]
                        groups = [g for g in groups if len(g) > 0]
                        
                        grand_mean = valid_data[num_col].mean()
                        ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                    else:
                        val = 0.0
                
                # Populate matrix mirror points symmetrically
                assoc_matrix.loc[col1, col2] = round(val, 3)
                assoc_matrix.loc[col2, col1] = round(val, 3)
                
        # 3. Plot the interactive Plotly Heatmap
        fig = px.imshow(
            assoc_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="Viridis",
            zmin=0.0, zmax=1.0,
            title="<b>Unified Absolute Association Matrix Heatmap (All Features)</b>",
            labels=dict(color="Association Strength")
        )
        fig.update_layout(width=max(650, n_cols * 50), height=max(550, n_cols * 50), template="plotly_white")
        fig.show()
        return assoc_matrix

    # =========================================================================
    # 3. MANOVA FIRST MOMENT HOMOGENEITY (WILKS' LAMBDA)
    # =========================================================================
    def test_constant_mean(self, columns: Optional[Sequence[str]] = None, chunks: int = 10):
        """
        Evaluates first moment homogeneity across sequential data blocks using MANOVA via Wilks' Lambda.
        Numerically stabilized via log-determinant tracking and shrinkage regularization.
        """
        if self.df is None:
            return print("❌ Error: No dataset available.")

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        # Drop rows with missing values locally to protect covariance calculations
        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        n, p = clean_df.shape

        if n < chunks * 2 or p == 0:
            return print(f"⚠️ Insufficient records ({n} rows, {p} features) to construct {chunks} discrete sequences.")

        # Split data array into uniform chunks
        data_blocks = np.array_split(clean_df.values, chunks, axis=0)
        grand_mean = clean_df.values.mean(axis=0)

        # Initialize matrices
        W = np.zeros((p, p)) # Within-class matrix
        B = np.zeros((p, p)) # Between-class matrix

        for block in data_blocks:
            b_len = len(block)
            if b_len == 0:
                continue
            b_mean = block.mean(axis=0)
            
            # Within-group calculation with Ledoit-Wolf shrinkage to ensure matrix stability
            diff_w = block - b_mean
            W += diff_w.T @ diff_w
            
            # Between-group calculation
            diff_b = b_mean - grand_mean
            B += b_len * np.outer(diff_b, diff_b)

        # Apply identity shrinkage regularization to avoid near-singular division blocks
        shrinkage = 1e-4
        W += np.eye(p) * shrinkage

        # Compute Wilks' Lambda via log-determinants for stability against underflow
        # Lambda = det(W) / det(W + B)
        try:
            sign_w, logdet_w = np.linalg.slogdet(W)
            sign_t, logdet_t = np.linalg.slogdet(W + B)
            
            wilks_lambda = np.exp(logdet_w - logdet_t)
            
            # Bartlett's approximation mapping Wilks' Lambda to a Chi-Squared metric
            df_degrees = p * (chunks - 1)
            bartlett_stat = -((n - 1) - (p + chunks) / 2.0) * (logdet_w - logdet_t)
            p_value = scipy.stats.chi2.sf(bartlett_stat, df_degrees)

            print("\n" + "="*65)
            print(" 🏛️ MANOVA STATIONARY FIRST MOMENT ANALYSIS (MEAN HOMOGENEITY)")
            print("="*65)
            print(f"🔹 Splitting Strategy   : {chunks} Sequential Block Intervals")
            print(f"🔹 Analyzed Subspace    : {p} Numeric Features across {n} total rows")
            print(f"🔹 Wilks' Lambda (Λ)   : {wilks_lambda:.6f}")
            print(f"🔹 Chi-Square Appx Stat : {bartlett_stat:.4f}")
            print(f"🔹 Computed p-value     : {p_value:.6e}")
            print("-"*65)
            if p_value < 0.05:
                print("🚨 Result: Reject Stationary Mean Hypothesis (p < 0.05).\n   Statistical mean drift or shift detected across chronological blocks.")
            else:
                print("✨ Result: Fail to Reject Stationary Mean Hypothesis.\n   The mean signature remains stable across consecutive blocks.")
            print("="*65 + "\n")
        except np.linalg.LinAlgError:
            print("❌ Matrix calculation error: Singularity error during log-determinant matrix analysis.")

    # =========================================================================
    # 4. BOX'S M VARIANCE HOMOGENEITY
    # =========================================================================
    def test_constant_covariance(self, columns: Optional[Sequence[str]] = None, chunks: int = 10):
        """
        Evaluates second moment homogeneity across sequential data blocks using Box's M-test.
        Numerically stabilized via shrinkage regularization and row-wise imputation protection.
        """
        if self.df is None:
            return print("❌ Error: No dataset available.")

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        n, p = clean_df.shape

        if n < chunks * p:
            return print(f"⚠️ Insufficient sample size ({n} rows) to accurately calculate independent {p}x{p} block covariance matrices.")

        data_blocks = np.array_split(clean_df.values, chunks, axis=0)
        
        # Calculate pooled covariance matrix
        pooled_cov = np.zeros((p, p))
        block_covs = []
        block_sizes = []
        
        shrinkage = 1e-4
        for block in data_blocks:
            n_k = len(block)
            if n_k <= 1:
                continue
            # Center the block matrix elements
            diff = block - block.mean(axis=0)
            cov_k = (diff.T @ diff) / (n_k - 1)
            # Inject shrinkage stability constraint
            cov_k += np.eye(p) * shrinkage
            
            block_covs.append(cov_k)
            block_sizes.append(n_k)
            pooled_cov += (n_k - 1) * cov_k

        N_minus_g = sum(n_k - 1 for n_k in block_sizes)
        pooled_cov /= N_minus_g

        # Compute Box's M value using log-determinants
        try:
            _, logdet_pooled = np.linalg.slogdet(pooled_cov)
            
            sum_m_term = 0.0
            for cov_k, n_k in zip(block_covs, block_sizes):
                _, logdet_k = np.linalg.slogdet(cov_k)
                sum_m_term += (n_k - 1) * logdet_k
                
            box_m = N_minus_g * logdet_pooled - sum_m_term
            
            # Calculate Scale Multipliers
            g = len(block_sizes)
            c1 = (((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * (sum(1.0 / (n_k - 1) for n_k in block_sizes) - (1.0 / N_minus_g)))
            
            # Chi-squared degree maps mapping formula
            df_m = (p * (p + 1) * (g - 1)) / 2.0
            stat_chi2 = box_m * (1.0 - c1)
            p_value = scipy.stats.chi2.sf(stat_chi2, df_m)

            print("\n" + "="*65)
            print(" 🏛️ BOX'S M TEST FOR HOMOGENEITY OF COVARIANCE MATRICES")
            print("="*65)
            print(f"🔹 Segment Subdivisions : {g} Sequential Block Paths")
            print(f"🔹 Box's M Metric       : {box_m:.4f}")
            print(f"🔹 Adjusted Chi-Square  : {stat_chi2:.4f}")
            print(f"🔹 Degrees of Freedom   : {df_m}")
            print(f"🔹 Calculated p-value   : {p_value:.6e}")
            print("-"*65)
            if p_value < 0.05:
                print("🚨 Result: Reject Covariance Stationarity (p < 0.05).\n   Significant variance or structural dispersion changes exist across rows.")
            else:
                print("✨ Result: Fail to Reject Covariance Stationarity.\n   Covariance dispersion matrices remain uniform across sequences.")
            print("="*65 + "\n")

        except np.linalg.LinAlgError:
            print("❌ Matrix operation failed: Singular structures encountered during Box M log-determinant analysis.")

    # =========================================================================
    # 5. MULTIVARIATE LJUNG-BOX PORTMANTEAU TEST
    # =========================================================================
    def test_row_independence(self, columns: Optional[Sequence[str]] = None, max_lag: Optional[int] = None):
        """
        Evaluates row-to-row statistical independence using the Multivariate Ljung-Box Portmanteau test.
        Numerically stabilized via pseudo-inverse fallbacks and row-wise missing value removal.
        """
        if self.df is None:
            return print("❌ Error: No dataset available.")

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        n, p = clean_df.shape

        if n < 5:
            return print("⚠️ Critical Error: Rows sample distribution is too shallow to perform lag calculations.")

        if max_lag is None:
            max_lag = int(np.floor(np.log(n)))
        max_lag = max(1, min(max_lag, n - 2))

        # Center our vector matrix elements
        data_matrix = clean_df.values - clean_df.values.mean(axis=0)
        
        # Calculate C(0) -> Zero Lag Baseline Covariance Matrix
        c_0 = (data_matrix.T @ data_matrix) / n
        # Use pseudo-inverse fallback to safely invert potential near-singular structures
        c_0_inv = np.linalg.pinv(c_0)

        # Accumulate cross-lag products trace sums
        q_stat = 0.0
        for lag in range(1, max_lag + 1):
            # Shift data structures to construct cross-lag pairs
            x_t = data_matrix[lag:]
            x_lag = data_matrix[:-lag]
            
            # Compute cross-covariance at current lag offset
            c_lag = (x_t.T @ x_lag) / n
            
            # Compute Portmanteau track metric trace: trace(C_lag^T * C_0^-1 * C_lag * C_0^-1)
            term = c_lag.T @ c_0_inv @ c_lag @ c_0_inv
            trace_val = np.trace(term)
            
            # Ljung-Box scaling multiplier
            q_stat += trace_val / (n - lag)
            
        q_stat *= (n * (n + 2))
        df_independence = (p ** 2) * max_lag
        p_value = scipy.stats.chi2.sf(q_stat, df_independence)

        print("\n" + "="*65)
        print(" 🏛️ MULTIVARIATE LJUNG-BOX TEST FOR SERIAL AUTOCORRELATION")
        print("="*65)
        print(f"🔹 Configured Row Offset Lag Limits : {max_lag}")
        print(f"🔹 Computed Portmanteau Q-Stat      : {q_stat:.4f}")
        print(f"🔹 Degrees of Freedom System Mapping: {df_independence}")
        print(f"🔹 System Convergence p-value       : {p_value:.6e}")
        print("-"*65)
        if p_value < 0.05:
            print("🚨 Result: Reject Row Independence Hypothesis (p < 0.05).\n   Strong row-to-row context dependencies or serial correlations detected.")
        else:
            print("✨ Result: Fail to Reject Row Independence Hypothesis.\n   Rows can be statistically handled as independent observations.")
        print("="*65 + "\n")

    # =========================================================================
    # 1. ESTIMATE PARAMETRIC JOINT NORMAL MICRO-SCALE MODEL (MLE)
    # =========================================================================
    def estimate_joint_normal(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Operationalizes the micro-scale model X_i ~ N(mu_hat, S) by fitting a 
        parametric Multivariate Normal Distribution to the verified IID baseline.
        Utilizes finite-sample unbiased Maximum Likelihood Estimation (MLE) with 
        Bessel's correction to construct a continuous probabilistic boundary.
        """
        if self.df is None:
            print("❌ Error: No dataset available.")
            return {}

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        # Isolate complete cases to verify mathematical integrity
        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        n, p = clean_df.shape

        if n <= p:
            print(f"⚠️ Dimensionality Constraint: Row count ({n}) must exceed feature count ({p}) for joint normal estimation.")
            return {}

        X = clean_df.values
        
        # 1. Unbiased Maximum Likelihood Estimation with Bessel's correction
        mu_hat = X.mean(axis=0)
        diff = X - mu_hat
        # Bessel's correction factor applied: 1 / (n - 1)
        S = (diff.T @ diff) / (n - 1)

        # 2. Construct continuous probabilistic boundary elements
        shrinkage = 1e-4
        S_stable = S + np.eye(p) * shrinkage
        
        try:
            # Instantiate the frozen distribution object from scipy
            mv_normal_dist = multivariate_normal(mean=mu_hat, cov=S_stable, allow_singular=True)
            
            # Calculate Mahalanobis distances to build real-time anomaly thresholds
            S_inv = np.linalg.pinv(S_stable)
            mahalanobis_distances = np.array([d @ S_inv @ d for d in diff])
            
            # Anomaly cutoffs using Chi-Square distribution critical bounds at 95% and 99%
            threshold_95 = scipy.stats.chi2.ppf(0.95, df=p)
            threshold_99 = scipy.stats.chi2.ppf(0.99, df=p)
            
            anomalies_count = np.sum(mahalanobis_distances > threshold_95)

            print("\n" + "="*65)
            print(" 📐 PARAMETRIC MULTIVARIATE NORMAL DISTRIBUTION ESTIMATION")
            print("="*65)
            print(f"🔹 Evaluated Features : {p} Numeric Space Nodes")
            print(f"🔹 Baseline Population: {n} Row Observations")
            print(f"🔹 Chi-Square Threshold (95% Boundary): {threshold_95:.4f}")
            print(f"🔹 Chi-Square Threshold (99% Boundary): {threshold_99:.4f}")
            print(f"🔹 Real-Time Outliers Flagged (>95% MD) : {anomalies_count} rows ({anomalies_count/n*100:.1f}%)")
            print("-"*65)

            return {
                "mean_vector": mu_hat,
                "covariance_matrix": S,
                "distribution_object": mv_normal_dist,
                "mahalanobis_distances": mahalanobis_distances,
                "threshold_95": threshold_95,
                "threshold_99": threshold_99
            }
        except Exception as e:
            print(f"❌ Distribution execution error: {e}")
            return {}

    # =========================================================================
    # 2. INSTANTIATE MACRO-SCALE CLT SAMPLING MATRIX
    # =========================================================================
    def instantiate_macro_clt_distribution(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Operationalizes the macro-scale CLT model: mu_hat_n ~ N(mu_hat, (1/n) * S).
        Instantiates the continuous Gaussian sampling distribution of the empirical mean vector
        to track system drift and draw confidence ellipsoids.
        """
        # Rely on the underlying micro-scale estimator framework
        mle_results = self.estimate_joint_normal(columns=columns)
        if not mle_results:
            return {}

        n = len(self.df[columns].dropna()) if columns else len(self.df.select_dtypes(include=[np.number]).dropna())
        mu_hat = mle_results["mean_vector"]
        S = mle_results["covariance_matrix"]
        p = len(mu_hat)

        # CLT scale adjustment: (1 / n) * Covariance Matrix
        clt_covariance = S / n
        shrinkage = 1e-5
        clt_cov_stable = clt_covariance + np.eye(p) * shrinkage

        try:
            clt_dist = multivariate_normal(mean=mu_hat, cov=clt_cov_stable, allow_singular=True)
            
            # Confidence Ellipsoid Scale Factor using Hotelling's T-Squared / F-Distribution maps
            # F_critical bound for mean vectors parameter space uncertainty evaluation
            f_crit = scipy.stats.f.ppf(0.95, df1=p, df2=n-p)
            hotelling_ellipsoid_scale = (p * (n - 1) / (n - p)) * f_crit

            print(" 🏛️ MACRO-SCALE CENTRAL LIMIT THEOREM (CLT) DISPATCHER")
            print("="*65)
            print(f"🔹 Sample Size (n)            : {n} Observations")
            print(f"🔹 Standard Error Vector Max  : {np.sqrt(np.diag(clt_covariance)).max():.6f}")
            print(f"🔹 Confidence Ellipsoid Radius: {hotelling_ellipsoid_scale:.4f}")
            print("✨ Continuous parameter variance distribution instantiated successfully.")
            print("="*65 + "\n")

            return {
                "clt_mean_vector": mu_hat,
                "clt_covariance_matrix": clt_covariance,
                "clt_distribution_object": clt_dist,
                "ellipsoid_scale_factor": hotelling_ellipsoid_scale
            }
        except Exception as e:
            print(f"❌ CLT Initialization Error: {e}")
            return {}

    # =========================================================================
    # 3. EMPIRICAL PRINCIPAL COMPONENT ANALYSIS (PCA Dashboard)
    # =========================================================================
    def compute_empirical_pca(self, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes geometric de-correlation framework via PCA. Decomposes unbiased S into orthogonal P_hat.
        Computes Hotelling's T^2 and Q (SPE) statistics across truncation boundaries k to optimize health monitoring thresholds.
        Generates an elite 2x3 Plotly Subplot Dashboard displaying structural metrics inline.
        """
        if self.df is None:
            print("❌ Error: No data loaded.")
            return {}

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        X = clean_df.values
        n, p = X.shape

        if n < 2 or p < 2:
            print("⚠️ Insufficient dimensions to decompose coordinate mappings.")
            return {}

        # Center and scale data matrix natively to standard normal ranges
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0, ddof=1)
        X_std[X_std == 0] = 1.0  # Safe division guard
        X_scaled = (X - X_mean) / X_std

        # Unbiased Empirical Correlation Matrix
        R = (X_scaled.T @ X_scaled) / (n - 1)

        # Eigenvalue Decomposition
        eigenvalues, P_hat = np.linalg.eigh(R)
        
        # Sort values in descending structural order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        P_hat = P_hat[:, idx]

        # Explained Variance Calculations
        total_var = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_var
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Precompute T^2 and Q Metrics for each Truncation Boundary Option k
        k_range = list(range(1, p + 1))
        t2_95_thresholds = []
        q_95_thresholds = []

        for k in k_range:
            # Hotelling T^2 critical bound via F-Distribution scaling
            t2_crit = (k * (n**2 - 1) / (n * (n - k))) * scipy.stats.f.ppf(0.95, k, n - k) if n > k else np.nan
            t2_95_thresholds.append(t2_crit)

            # Q (SPE) Threshold Calculation using Box's approximate multi-weighting mapping formula
            if k < p:
                theta1 = np.sum(eigenvalues[k:])
                theta2 = np.sum(eigenvalues[k:]**2)
                theta3 = np.sum(eigenvalues[k:]**3)
                h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2**2)
                
                if h0 <= 0: h0 = 1.0 # Protect root transformations
                ca = scipy.stats.norm.ppf(0.95)
                q_crit = theta1 * ((ca * np.sqrt(2.0 * theta2 * h0**2) / theta1) + 1.0 + (theta2 * h0 * (h0 - 1.0)) / (theta1**2))**(1.0 / h0)
            else:
                q_crit = 0.0
            q_95_thresholds.append(q_crit)

        if show_plot:
            # 2x3 Plotly Grid Blueprint Dashboard Setup
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "Scree Plot (Eigenvalues)", "Individual Variance Profile", "Cumulative Energy Profile",
                    "Feature Loadings Matrix (PC1 vs PC2)", "Hotelling T² Boundary Threshold", "Q-Statistic (SPE) Fence Line"
                ),
                horizontal_spacing=0.08, vertical_spacing=0.15
            )

            # 1. Scree Plot Trace
            fig.add_trace(go.Bar(x=[f"PC{i}" for i in k_range], y=eigenvalues, marker_color="royalblue", name="Eigenvalue"), row=1, col=1)
            # 2. Individual Variance
            fig.add_trace(go.Scatter(x=[f"PC{i}" for i in k_range], y=explained_variance_ratio, mode="markers+lines", marker=dict(color="teal"), name="Indiv Var"), row=1, col=2)
            # 3. Cumulative Energy
            fig.add_trace(go.Scatter(x=[f"PC{i}" for i in k_range], y=cumulative_variance_ratio, mode="markers+lines+text", text=[f"{v*100:.0f}%" for v in cumulative_variance_ratio], textposition="bottom right", marker=dict(color="crimson"), name="Cumul Var"), row=1, col=3)
            
            # 4. Feature Loading Scatter Map
            fig.add_trace(go.Scatter(x=P_hat[:, 0], y=P_hat[:, 1], mode="markers+text", text=target_cols, textposition="top center", marker=dict(size=10, color="purple"), name="Loadings"), row=2, col=1)
            # Add center crosshairs to loading plot
            fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(dash="dash", color="grey", width=1), row=2, col=1)
            fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(dash="dash", color="grey", width=1), row=2, col=1)
            
            # 5. T2 Critical Bounds
            fig.add_trace(go.Scatter(x=k_range, y=t2_95_thresholds, mode="markers+lines", marker=dict(color="darkorange"), name="T² Threshold"), row=2, col=2)
            # 6. Q Critical Bounds
            fig.add_trace(go.Scatter(x=k_range, y=q_95_thresholds, mode="markers+lines", marker=dict(color="forestgreen"), name="Q Threshold"), row=2, col=3)

            fig.update_layout(height=720, width=1200, title_text="<b>Empirical Principal Component Analysis (PCA) Dashboard</b>", showlegend=False, template="plotly_white")
            fig.update_yaxes(title_text="Magnitude", row=1, col=1)
            fig.update_yaxes(title_text="Ratio (0-1)", row=1, col=2)
            fig.update_yaxes(title_text="Ratio (0-1)", row=1, col=3)
            fig.update_xaxes(title_text="PC1 Structural Coefficient", row=2, col=1)
            fig.update_yaxes(title_text="PC2 Structural Coefficient", row=2, col=1)
            fig.update_xaxes(title_text="Truncation Boundary (k)", row=2, col=2)
            fig.update_yaxes(title_text="Critical Limit Value", row=2, col=2)
            fig.update_xaxes(title_text="Truncation Boundary (k)", row=2, col=3)
            fig.update_yaxes(title_text="Critical Limit Value", row=2, col=3)
            fig.show()

        return {
            "eigenvalues": eigenvalues,
            "loading_vectors": P_hat,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": cumulative_variance_ratio,
            "t2_thresholds": t2_95_thresholds,
            "q_thresholds": q_95_thresholds
        }

    # =========================================================================
    # 4. EMPIRICAL FACTOR ANALYSIS SUBSPACE FRAMEWORK (FA Dashboard)
    # =========================================================================
    def compute_empirical_fa(self, k: int, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes the Factor Analysis latent subspace framework.
        Decomposes R into shared structural variance (Lambda * Lambda^T) and unique variance (Psi).
        Estimates latent common factor scores using Thomson's MMSE regression method.
        Generates an elite 4-panel Plotly Subplot Dashboard displaying extraction patterns inline.
        """
        if self.df is None:
            print("❌ Error: No dataset available.")
            return {}

        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in target_cols:
            target_cols.remove('count')

        clean_df = self.df[target_cols].dropna().reset_index(drop=True)
        X = clean_df.values
        n, p = X.shape

        if k >= p:
            print(f"⚠️ Truncation Bound Failure: Factor components selection count k={k} must be strictly lower than total features count p={p}.")
            return {}

        # Standardizing matrices natively
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        R = (X_scaled.T @ X_scaled) / (n - 1)

        try:
            # Leverage scikit-learn's underlying expectation maximization routines
            fa = FactorAnalysis(n_components=k, rotation=None, max_iter=1000, tol=1e-4)
            fa.fit(X_scaled)
            
            # Extract Factor Loadings (Matrix Lambda)
            # Transposed to align rows as features and columns as latent factor components
            lambda_matrix = fa.components__.T 
            
            # Extract sensor noise/uniqueness scales diagonal vector elements (Psi Matrix)
            psi_vector = fa.noise_variance_
            
            # Communality calculation: shared variance captured per feature variable
            communality_vector = np.sum(lambda_matrix**2, axis=1)
            
            # Calculate Latent Factor Energy profiles
            factor_energy = np.sum(lambda_matrix**2, axis=0)

            # Thomson's MMSE Regression Method to construct latent factor scores
            # Factor Score Coefficients = R^-1 @ Lambda
            R_inv = np.linalg.pinv(R)
            thomson_weights = R_inv @ lambda_matrix
            factor_scores = X_scaled @ thomson_weights

            if show_plot:
                # 4-Panel Plotly Matrix Grid Initialization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Latent Subspace Factor Loadings (Matrix Λ)",
                        "Communality vs Uniqueness Variance Balancing",
                        "Sensor Noise Scale Dispersion Profile (Ψ Matrix)",
                        "Extracted Latent Factors Energy Signature"
                    ),
                    horizontal_spacing=0.12, vertical_spacing=0.22
                )

                # 1. Heatmap for Lambda Matrix Loadings
                fig.add_trace(
                    go.Heatmap(
                        z=lambda_matrix, x=[f"Factor {i+1}" for i in range(k)], y=target_cols,
                        colorscale="Viridis", colorbar=dict(title="Loading Scale", x=0.45, y=0.78, len=0.4)
                    ),
                    row=1, col=1
                )

                # 2. Grouped Stacked Bar for Variance Balance Profiles
                fig.add_trace(go.Bar(y=target_cols, x=communality_vector, name="Communality (Shared)", orientation='h', marker_color="mediumseagreen"), row=1, col=2)
                fig.add_trace(go.Bar(y=target_cols, x=psi_vector, name="Uniqueness (Specific Noise)", orientation='h', marker_color="lightcoral"), row=1, col=2)

                # 3. Scatter Line for Sensor Uniqueness Noise Dispersions
                fig.add_trace(go.Scatter(x=target_cols, y=psi_vector, mode="markers+lines", marker=dict(size=9, color="darkmagenta"), name="Noise Scale"), row=2, col=1)
                
                # 4. Bar Chart for Individual Energy Signatures
                fig.add_trace(go.Bar(x=[f"Factor {i+1}" for i in range(k)], y=factor_energy, marker_color="chocolate", name="Energy Scale"), row=2, col=2)

                fig.update_layout(height=720, width=1200, title_text=f"<b>Empirical Latent Factor Analysis Subspace Dashboard (k={k})</b>", barmode="stack", showlegend=False, template="plotly_white")
                fig.update_xaxes(title_text="Latent Dimensions Path", row=1, col=1)
                fig.update_xaxes(title_text="Variance Proportions (0-1 Range)", row=1, col=2)
                fig.update_xaxes(title_text="Analyzed Input Features", row=2, col=1)
                fig.update_yaxes(title_text="Unique Noise Scale Value", row=2, col=1)
                fig.update_xaxes(title_text="Extracted Subspace Factors", row=2, col=2)
                fig.update_yaxes(title_text="Eigenvalue Equivalent Energy", row=2, col=2)
                fig.show()

            return {
                "factor_loadings": lambda_matrix,
                "uniqueness_variance": psi_vector,
                "communalities": communality_vector,
                "factor_energy_magnitudes": factor_energy,
                "thomson_regression_weights": thomson_weights,
                "latent_factor_scores": factor_scores
            }
        except Exception as e:
            print(f"❌ Subspace extraction routine execution error: {e}")
            return {}
        
    
        
    ''' def extract_normalized_categorical_data(self, method='uniform'):
        if self.df is None: return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()

        if cat_df.empty:
            print("⚠️ No categorical columns found to transform.")
            self.categorical_normalized_df = pd.DataFrame()
            return self.categorical_normalized_df

        method_lower = method.lower().strip()
        if method_lower == 'uniform':
            for col in cat_df.columns:
                codes = cat_df[col].astype('category').cat.codes
                max_code = codes.max()
                cat_df[col] = codes / max_code if max_code > 0 else 0.0
            self.categorical_normalized_df = cat_df
        elif method_lower == 'ordinal':
            encoder = OrdinalEncoder()
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=cat_df.columns, index=cat_df.index)
        elif method_lower == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            feature_names = encoder.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded_data, columns=feature_names, index=cat_df.index)
        elif method_lower == 'minmax_ordinal':
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            cat_df_filled = cat_df.fillna("Missing")
            encoded_data = encoder.fit_transform(cat_df_filled)
            scaled_data = scaler.fit_transform(encoded_data)
            self.categorical_normalized_df = pd.DataFrame(scaled_data, columns=cat_df.columns, index=cat_df.index)
        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method='uniform')

        print(f"✨ Successfully encoded categorical data using the '{method_lower}' method.")
        return self.categorical_normalized_df     

    def create_normalized_data_df(self):
        if self.df is None: return print("Error: No data loaded.")
        num_df = self.extract_numeric_data()
        cat_norm_df = self.extract_normalized_categorical_data()

        if cat_norm_df is None or cat_norm_df.empty:
            self.normalized_data_df = num_df
            return self.normalized_data_df
        if num_df is None or num_df.empty:
            self.normalized_data_df = cat_norm_df
            return self.normalized_data_df

        self.normalized_data_df = pd.concat([num_df, cat_norm_df], axis=1)
        print(f"✅ Success! Created merged DataFrame with {self.normalized_data_df.shape[1]} columns.")
        return self.normalized_data_df

    def plot_numerical(self, column_names):
        if self.df is None: return
        if isinstance(column_names, str): column_names = [column_names]
        valid_cols = [c for c in column_names if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]

        for col in valid_cols:
            fig = make_subplots(rows=1, cols=3, subplot_titles=(f"Horizontal Violin/Box: {col}", f"Scatter Plot: {col}", f"Distribution: {col}"))
            fig.add_trace(go.Violin(x=self.df[col], box_visible=True, meanline_visible=True, name=col, orientation='h', line_color='lightseagreen'), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.df[col], mode='markers', marker=dict(opacity=0.5, color='royalblue'), name=col), row=1, col=2)
            fig.add_trace(go.Histogram(x=self.df[col], name=col, marker_color='indianred'), row=1, col=3)
            fig.update_layout(height=450, title_text=f"<b>Statistical Analysis: {col}</b>", showlegend=False, template="plotly_white")
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_xaxes(title_text="Value", row=1, col=3)
            fig.show()

    def plot_categorical(self, column_names):
        if self.df is None: return
        if isinstance(column_names, str): column_names = [column_names]

        for col in column_names:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(1).astype(str) + '%'
            fig = px.bar(counts, x=col, y='count', text='percentage', title=f"Frequency: {col}", color=col, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.show()

    def handle_outliers(self, columns=None, find_and_delete=False):
        if self.df is None: return
        target_cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_outliers = set()

        for col in target_cols:
            if col == 'count': continue
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))]
            all_outliers.update(outliers.index.tolist())
            print(f"🚨 {col}: Found {len(outliers)} outliers.")

        if all_outliers:
            display(self.df.loc[list(all_outliers)])
            if find_and_delete:
                self.df = self.df.drop(index=list(all_outliers)).reset_index(drop=True)
                print(f"🗑️ Deleted {len(all_outliers)} outlier rows.")

    def plot_relationship(self, col1, col2):
        if self.df is None: return
        is_num1, is_num2 = pd.api.types.is_numeric_dtype(self.df[col1]), pd.api.types.is_numeric_dtype(self.df[col2])

        if is_num1 and is_num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Correlation: {col1} vs {col2}")
        elif is_num1 != is_num2:
            num, cat = (col1, col2) if is_num1 else (col2, col1)
            fig = px.box(self.df, x=cat, y=num, points="all", color=cat, title=f"Distribution of {num} by {cat}")
        else:
            fig = px.histogram(self.df, x=col1, color=col2, barmode="group", title=f"Relationship: {col1} vs {col2}")
        fig.show()

    def plot_numerical_correlation(self):
        if self.df is None: return
        numerical_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'count' in numerical_df.columns:
            numerical_df = numerical_df.drop(columns=['count'])
        corr = numerical_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Pearson Correlation Heatmap")
        fig.show()

    def plot_categorical_correlation(self):
        if self.df is None: return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("⚠️ No categorical columns found to compute associations.")
            
        cols = cat_df.columns
        n_cols = len(cols)
        corr_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
        
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                    continue
                    
                confusion_matrix = pd.crosstab(cat_df[col1], cat_df[col2])
                if confusion_matrix.size == 0 or min(confusion_matrix.shape) <= 1:
                    continue
                    
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                v = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1))) if n > 0 else 0.0
                corr_matrix.loc[col1, col2] = v
                corr_matrix.loc[col2, col1] = v
                
        print("--- Cramér's V Association Matrix ---")
        display(corr_matrix.round(3))
        
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="<b>Cramér's V Categorical Association Heatmap</b>", labels=dict(color="Cramér's V"))
        fig.update_layout(height=max(400, n_cols * 80), width=max(500, n_cols * 80), template="plotly_white")
        fig.show()
        return corr_matrix

    def plot_all_associations_heatmap(self):
        if self.df is None: return print("Error: No data loaded.")
        cols = [c for c in self.df.columns if c != 'count']
        n_cols = len(cols)
        assoc_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
        
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue
                
                valid_data = self.df[[col1, col2]].dropna()
                if valid_data.empty: continue
                
                is_num1 = pd.api.types.is_numeric_dtype(valid_data[col1])
                is_num2 = pd.api.types.is_numeric_dtype(valid_data[col2])
                
                if is_num1 and is_num2:
                    val = abs(valid_data[col1].corr(valid_data[col2], method='pearson'))
                elif not is_num1 and not is_num2:
                    confusion_matrix = pd.crosstab(valid_data[col1], valid_data[col2])
                    if confusion_matrix.size > 0 and min(confusion_matrix.shape) > 1:
                        chi2 = chi2_contingency(confusion_matrix)[0]
                        n = confusion_matrix.sum().sum()
                        val = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1))) if n > 0 else 0.0
                    else:
                        val = 0.0
                else:
                    cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
                    categories = valid_data[cat_col].unique()
                    if len(categories) > 1:
                        groups = [valid_data[valid_data[cat_col] == c][num_col] for c in categories if len(valid_data[valid_data[cat_col] == c][num_col]) > 0]
                        grand_mean = valid_data[num_col].mean()
                        ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                    else:
                        val = 0.0
                
                assoc_matrix.loc[col1, col2] = round(val, 3)
                assoc_matrix.loc[col2, col1] = round(val, 3)
                
        print("--- Global Association Matrix ---")
        display(assoc_matrix)
        
        fig = px.imshow(assoc_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="viridis", title="<b>Unified Association Heatmap (Numeric & Categorical)</b>", labels=dict(color="Association Strength"))
        fig.update_layout(height=max(500, n_cols * 45), width=max(600, n_cols * 45), template="plotly_white")
        fig.show()
        return assoc_matrix  '''

import time
from datetime import datetime
import uuid
import json
import pandas as pd
import numpy as np
import copy
import inspect
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.io import to_html
import networkx as nx
import graphviz
import base64
from IPython.display import display, HTML

class PlottingMethods:

    def get_methods_info(self, user_id=None):
        method_dicts = []
        # Introspect class methods
        methods = inspect.getmembers(self, inspect.ismethod)

        for name, method in methods:
            # Skip private methods
            if name.startswith('_'):
                continue

            # Get method signature
            signature = inspect.signature(method)

            # Get docstring
            docstring = method.__doc__
            formatted_docstring = docstring.strip() if docstring else "No description available"

            # Append method documentation
            method_dicts += [{"method": name, "signature": str(signature), "description": formatted_docstring}]
            
        status = 'success'
        return {'status': status, 'response': method_dicts}

    def _data_validate(self, data, message_dict):
        """
        Validates the structure of the input data.
        Handles missing inputs, DataFrames, list records, dict payloads, and raw JSON strings.
        Returns a standardized response dict containing status and data.
        """
        if data is None:
            raise ValueError("Input data cannot be empty or None.")

        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("The provided DataFrame is empty.")
            return {'status': 'success', 'data': data.copy()}

        if isinstance(data, list):
            if not data:
                raise ValueError("The provided list of records is empty.")
            return {'status': 'success', 'data': pd.DataFrame(data)}

        if isinstance(data, dict):
            if "records" in data:
                records = data["records"]
                if isinstance(records, dict):
                    records = [records]
                if not isinstance(records, list):
                    raise ValueError("The extracted JSON records block must be a list of objects.")
                if not records:
                    raise ValueError("The extracted JSON records block is empty.")
                return {'status': 'success', 'data': pd.DataFrame(records)}

            if not data:
                raise ValueError("The provided dictionary data is empty.")
            return {'status': 'success', 'data': pd.DataFrame([data])}

        if isinstance(data, str):
            try:
                parsed_json = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON string: {str(e)}")

            if isinstance(parsed_json, dict) and "records" in parsed_json:
                records = parsed_json["records"]
            elif isinstance(parsed_json, list):
                records = parsed_json
            elif isinstance(parsed_json, dict):
                if not parsed_json:
                    raise ValueError("The provided JSON object is empty.")
                return {'status': 'success', 'data': pd.DataFrame([parsed_json])}
            else:
                raise ValueError("JSON string must be a raw list of objects or a dict containing a 'records' key.")

            if isinstance(records, dict):
                records = [records]
            if not records:
                raise ValueError("The extracted JSON records block is empty.")
            return {'status': 'success', 'data': pd.DataFrame(records)}

        raise ValueError(f"Unsupported data type structure: {type(data)}")

    def plot_bar_chart(self, x='date', y='value', color=None, text=None, title='', barmode='stack', hover_data=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px bar chart with x as the x values,
        y as the y values, and color as the categories. The variable barmode specifies if the mode of grouping.

        Args:
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            color (str): Optional - Column name for the stacking categories.
            text (str or None): Optional - Column name for text labels.
            title (str): Optional - Title of the chart.
            barmode (str or None): Optional - The bar mode either stack or group.
            hover_data (list or None): Optional - List of column names to include in hover data.
            data (str): JSON string containing a list of records in the format
                        {'records': [{'x': ..., 'y': ..., 'color': ...}, ...]}.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')

            if isinstance(hover_data, str):
                try:
                    # First, attempt to parse it as JSON
                    parsed_data = json.loads(hover_data)

                    # If parsing succeeds and the result is a list, use it
                    if isinstance(parsed_data, list):
                        hover_data = parsed_data
                    else:
                        # If it's valid JSON but not a list, set hover_data to None or handle as needed
                        hover_data = None
                except json.JSONDecodeError:
                    # If JSON parsing fails, check if it's a comma-separated string
                    if ',' in hover_data:
                        hover_data = hover_data.split(',')
                    else:
                        # Handle non-JSON, non-comma-separated strings
                        hover_data = None

            include_plotlyjs = True
            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[y] = pd.to_numeric(df[y])
            c_categories_labels = None
            if color is not None:
                df.dropna(subset=[color], inplace=True)
                # Use unique() and sort if necessary
                c_categories_labels = df[color].unique()
                if not any(sub in color.lower() for sub in ['month', 'week']):
                    c_categories_labels = sorted(c_categories_labels)
                df[color] = pd.Categorical(df[color], categories=c_categories_labels, ordered=True)

            # Ensure xLabel sorting
            x_categories_labels = df[x].unique()
            df[x] = pd.Categorical(df[x], categories=x_categories_labels, ordered=True)

            if hover_data:
                hover_data = [col for col in hover_data if col in df.columns]

            # Generate a stacked bar graph
            cat_orders = {x: x_categories_labels}

            # 2. Only add color to category_orders if color is actually provided
            if color is not None and c_categories_labels is not None:
                cat_orders[color] = c_categories_labels

            # 3. Generate a stacked bar graph using the dynamic dictionary
            fig = px.bar(
                df,
                x=x,
                y=y,
                color=color,
                title=title,
                text=text,
                hover_data=hover_data,
                category_orders=cat_orders  # Use the cleaned dictionary
            )

            # Customize the layout
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y,
                uniformtext_minsize=8,
                uniformtext_mode='hide',  # Hide overlapping text
                barmode=barmode           # Enable barmode configuration configuration
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness here
                include_plotlyjs=include_plotlyjs
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Return the response with the figure
            message_dict.update({'message': 'Bar chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
        
    def plot_pie_chart(self, names='date', values='value', title='', hole=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates a responsive Plotly pie chart based on provided or previously stored data.

        This method creates a pie chart using the Plotly Express library. Data can be provided directly
        via a JSON string (`data`) or indirectly through a reference ID (`data_id`).

        Parameters:
            names (str): Column name in the dataset to use for pie chart segment labels (default: 'date').
            values (str): Column name in the dataset to use for segment sizes (default: 'value').
            title (str): Title of the pie chart (default: empty string).
            data_id (str, optional): Optional ID for retrieving stored data via `DBQ.get_ai_message_stored_data`.
            data (str, optional): JSON string in the format `{"records": [...]}` representing the dataset.
                                  Used if `data_id` is not provided.
            meta_data (dict): Additional metadata dictionary to include in the response.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if the chart is generated; 'error' if there is an issue.
                - 'response' (dict):
                    - 'meta_data' (dict): Includes original meta plus status or error message.
                    - 'data' (str): JSON-encoded string containing the HTML representation of the pie chart.
                    - 'message' (str): A stringified version of `meta_data` for convenience.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[values] = pd.to_numeric(df[values])

            # Generate an interactive pie chart
            fig = px.pie(
                df,
                names=names,
                values=values,
                title=title if title else None,
                hole=hole
            )

            # Uniform text and margin constraints configuration matching standard layout definitions
            fig.update_layout(
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness and strip brand marks
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Pie chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_histogram(self, x='value', title='', bins=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px histogram with the specified column as the x-axis.

        Args:
            x (str): Column name for the x-axis.
            title (str): Title of the histogram.
            bins (list or None): Custom bin intervals for the histogram.
            data (str): JSON string containing a list of records.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[x] = pd.to_numeric(df[x], errors='coerce')

            # Calculate the number of bins if a list of custom bin boundaries is provided
            nbins = len(bins) - 1 if isinstance(bins, (list, np.ndarray)) and len(bins) > 1 else None

            # Generate the histogram graph
            fig = px.histogram(
                df,
                x=x,
                title=title if title else None,
                nbins=nbins
            )

            # If explicit bin array boundaries are passed, apply precise adjustments to the xbins traces
            if nbins and hasattr(fig.data[0], 'xbins'):
                fig.update_traces(xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]))

            # Customize layout constraints
            fig.update_layout(
                xaxis_title=x,
                yaxis_title="Count",
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                template="plotly_white"
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness and strip brand marks
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Histogram plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_simple_sunburst_graph(self, path=["parent", "name"], values="marks", title='Hierarchy map', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plots a Plotly Sunburst graph given hierarchical data.

        Args:
            data (str): JSON string of hierarchical records.
            path (list): List defining the hierarchy in the dataset.
            values (str): Column name to use for slice sizes.
            title (str): Title of the Sunburst plot.

        Returns:
            dict: A response containing the Plotly figure.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[values] = pd.to_numeric(df[values], errors='coerce')

            # Ensure all hierarchy levels in the path are clean string categories
            if isinstance(path, list):
                for col in path:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

            # Generate the sunburst chart
            fig = px.sunburst(
                df,
                path=path,
                values=values,
                title=title if title else None
            )

            # Standardize layout constraints
            fig.update_layout(
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                template="plotly_white"
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness and strip brand marks
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Sunburst graph plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_sankey_diagram(self, source_column="parent", target_column="name", values="marks", title="Sankey Diagram", data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plot a Plotly Sankey diagram using the specified source and target columns.

        Args:
            source_column (str): The column name to use as the source in the Sankey diagram.
            target_column (str): The column name to use as the target in the Sankey diagram.
            values (str): The column name to use as the values in the Sankey diagram.
            title (str): The title of the Sankey diagram.
            data (str): JSON string containing the records of hierarchical data.

        Returns:
            dict: A dictionary containing:
                  - status (str): 'success' or 'error'.
                  - response (dict):
                      - message (str): Success or error message.
                      - data (dict): Contains the Sankey diagram figure as a Plotly figure object.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data to a pandas DataFrame
            df = pd.DataFrame(data)
            df[values] = pd.to_numeric(df[values], errors='coerce')
            
            # Drop rows with missing crucial links or weights
            df.dropna(subset=[source_column, target_column, values], inplace=True)

            # Map unique string labels to unique 0-based integer positions for Plotly's Sankey engine
            all_nodes = sorted(list(set(df[source_column].astype(str)).union(set(df[target_column].astype(str)))))
            node_indices = {node: idx for idx, node in enumerate(all_nodes)}

            sources_mapped = df[source_column].astype(str).map(node_indices).tolist()
            targets_mapped = df[target_column].astype(str).map(node_indices).tolist()
            values_list = df[values].tolist()

            # Generate the low-level Graph Object Sankey structure
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="rgba(31, 119, 180, 0.8)"  # Standard aesthetic slate blue
                ),
                link=dict(
                    source=sources_mapped,
                    target=targets_mapped,
                    value=values_list
                )
            )])

            # Apply layout choices
            fig.update_layout(
                title_text=title if title else None,
                font_size=10,
                template="plotly_white"
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},  # Enable responsiveness and strip brand marks
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Sankey diagram plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_sunburst_from_hierarchy(self, path=["parent", "name"], values="marks", title="Sunburst Diagram", data_id=None, data="{'records':[]}", meta_data={}, user_id=None):
        """
        Generate a Plotly Sunburst chart from hierarchical data.
        """
        message_dict = {'message': meta_data}
        validated_response = self._data_validate(data, message_dict)

        if not validated_response.get('status') == 'success':
            message_dict = validated_response.get('message_dict', {})
            message = message_dict.get('message', 'Error passing data')
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
        
        data = validated_response.get('data')
        include_plotlyjs = True

        # Inner helper to build the hierarchical DataFrame with aggregated marks
        def minimal_sunburst_hierarchy(
            data_payload: str,
            path_cols: list = ["parent", "name"],
            values_col: str = "marks",
            top_level: str = None,
            default_root: str = "Root"
        ) -> pd.DataFrame:
            """
            Build a hierarchical DataFrame for a Sunburst chart with aggregated marks.

            Args:
                data_payload (str): JSON string containing hierarchical records.
                path_cols (list): Columns specifying parent-child relationships.
                values_col (str): Column specifying values (marks) for the chart.
                top_level (str or None): Name of the top-level node in the hierarchy. If None, it will be inferred.
                default_root (str): Default root node name if multiple top-level nodes are detected.

            Returns:
                pd.DataFrame: Hierarchical DataFrame with aggregated marks for all nodes.
            """
            # Parse input data
            try:
                # Handle cases where data validation returned a DataFrame/dict vs raw JSON string
                if isinstance(data_payload, pd.DataFrame):
                    df_inner = data_payload.copy()
                elif isinstance(data_payload, list):
                    df_inner = pd.DataFrame(data_payload)
                elif isinstance(data_payload, dict) and "records" in data_payload:
                    df_inner = pd.DataFrame(data_payload["records"])
                else:
                    parsed = json.loads(data_payload)
                    records = parsed.get("records", []) if isinstance(parsed, dict) else parsed
                    if not records:
                        raise ValueError("Input data is empty or invalid.")
                    df_inner = pd.DataFrame(records)
            except Exception as e:
                raise ValueError(f"Failed to parse data: {e}")

            # Clean and prepare the DataFrame
            df_inner[path_cols[0]] = df_inner[path_cols[0]].replace([None, '', pd.NA], default_root).astype(str).str.strip()
            df_inner[path_cols[1]] = df_inner[path_cols[1]].astype(str).str.strip()
            df_inner[values_col] = pd.to_numeric(df_inner[values_col], errors='coerce')

            # Infer the top-level node if not provided
            if top_level is None:
                potential_top_levels = set(df_inner[path_cols[0]].unique()) - set(df_inner[path_cols[1]].unique())
                top_level = potential_top_levels.pop() if len(potential_top_levels) == 1 else default_root

            # Ensure the top-level node exists in the DataFrame
            if top_level not in df_inner[path_cols[0]].values:
                new_row = pd.DataFrame({path_cols[0]: [default_root], path_cols[1]: [top_level], values_col: [None]})
                df_inner = pd.concat([new_row, df_inner], ignore_index=True)

            # Build the hierarchy dictionary
            def build_hierarchy(df_target: pd.DataFrame, target_path: list) -> dict:
                hierarchy = {}
                for _, row in df_target.iterrows():
                    parent, child = row[target_path[0]], row[target_path[1]]
                    hierarchy.setdefault(parent, []).append(child)
                return hierarchy

            hierarchy_dict = build_hierarchy(df_inner, path_cols)

            # Aggregate marks for intermediate levels
            def aggregate_marks(node_name: str, hierarchy: dict, values_map: dict) -> float:
                if node_name in values_map and pd.notnull(values_map[node_name]):
                    return values_map[node_name]
                if node_name in hierarchy:
                    child_sum = sum(aggregate_marks(child, hierarchy, values_map) for child in hierarchy[node_name])
                    values_map[node_name] = child_sum
                    return child_sum
                return 0

            values_dict = {row[path_cols[1]]: row[values_col] for _, row in df_inner.iterrows() if pd.notnull(row[values_col])}
            for node in hierarchy_dict:
                aggregate_marks(node, hierarchy_dict, values_dict)

            # Expand the hierarchy into linear paths
            def expand_hierarchy(node_name: str, hierarchy: dict, current_path: list) -> list:
                current_path.append(node_name)
                if node_name in hierarchy:
                    nested_paths = []
                    for child in hierarchy[node_name]:
                        nested_paths.extend(expand_hierarchy(child, hierarchy, current_path.copy()))
                    return nested_paths
                return [current_path]

            all_paths = expand_hierarchy(top_level, hierarchy_dict, [])

            # Construct the hierarchical DataFrame output matrix
            max_depth = max(len(p) for p in all_paths) if all_paths else 0
            hierarchical_data = []
            for p in all_paths:
                row_dict = {f"level_{i}": (p[i] if i < len(p) else None) for i in range(max_depth)}
                row_dict[values_col] = values_dict.get(p[-1], 0)
                hierarchical_data.append(row_dict)

            return pd.DataFrame(hierarchical_data)

        try:
            # Parse and construct the hierarchy using the internal helper function
            hierarchy_results = minimal_sunburst_hierarchy(data, path_cols=path, values_col=values)
            hierarchical_df = hierarchy_results

            # Determine path columns for the Sunburst chart dynamically (excluding values column)
            path_columns = [col for col in hierarchical_df.columns if col != values and hierarchical_df[col].notnull().any()]

            # Generate the Sunburst chart
            fig = px.sunburst(
                hierarchical_df,
                path=path_columns,
                values=values,
                title=title if title else None
            )

            # Customize text options
            fig.update_layout(
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                template="plotly_white"
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Formulate valid metadata payloads matching structural requirements
            message_dict.update({'message': 'Sunburst chart plotted'})
            message = json.dumps(message_dict)
            response = {
                'status': 'success', 
                'response': {
                    'meta_data': message_dict, 
                    'data': json.dumps({'figure': fig_return}), 
                    'message': message
                }
            }
            return response

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            return {
                'status': 'error', 
                'response': {
                    'meta_data': message_dict, 
                    'data': json.dumps({'figure': ''}), 
                    'message': message
                }, 
                'message': f'Error: {str(e)}'
            }
    def plot_heat_map(self, values='Sales', index='Region', columns='Category', aggregade_method='sum', fill_value=0, title='Heatmap of Normalized Marks', width=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates an interactive Plotly heatmap from tabular data, with optional aggregation and layout control.

        This method builds a heatmap visualization using a pivoted version of the input dataset.
        Data can be provided either directly (`data`) or fetched via a reference `data_id`.

        Parameters:
            values (str): The column whose values will be visualized in the heatmap (e.g., grades, sales).
            index (str): The column to use as the y-axis in the pivot table.
            columns (str): The column to use as the x-axis in the pivot table.
            aggregade_method (str): Aggregation method to apply when pivoting ('sum', 'mean', 'count', etc.). Defaults to 'sum'.
            fill_value (int or float): Value used to fill missing cells in the pivot table. Defaults to 0.
            title (str): Title of the heatmap.
            width (int, optional): Optional fixed width of the chart in pixels. If None, it's auto-sized.
            data (str, optional): A JSON string in the format `{"records": [...]}`.
            meta_data (dict): Metadata dictionary to propagate through the response.

        Returns:
            dict: A dictionary containing status, response payload, and serialized message strings.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data payload into a processing DataFrame
            df = pd.DataFrame(data)
            df[values] = pd.to_numeric(df[values], errors='coerce')

            # Build matrix using pivot table calculations
            pivot_df = df.pivot_table(
                values=values,
                index=index,
                columns=columns,
                aggfunc=aggregade_method,
                fill_value=fill_value
            )

            # Generate the Heatmap using imshow engine
            fig = px.imshow(
                pivot_df,
                labels=dict(x=columns, y=index, color=values),
                x=pivot_df.columns.astype(str).tolist(),
                y=pivot_df.index.astype(str).tolist(),
                title=title if title else None,
                text_auto=True  # Display values inside cells dynamically
            )

            # Enforce layout configurations
            fig_layout = dict(
                template="plotly_white",
                xaxis=dict(type='category'),
                yaxis=dict(type='category')
            )
            if width:
                fig_layout['width'] = width
                
            fig.update_layout(**fig_layout)

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Heat map plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_multi_column_bar_graph(self, xLabel="Week", value_vars=['Total Slots Allocated by Region', 'Containers Product Ordered by Region'], title="Slot Allocation", orientation='v', hover_data=[], barmode='group', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plots a Plotly multi-column bar graph by melting multiple value columns 
        into a long-form dataset and visualizing them as grouped or stacked bars.
        """
        try:
            message_dict = {'message': meta_data}
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            data = validated_response.get('data')
            include_plotlyjs = True

            # Convert JSON data payload into a processing DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                message_dict.update({'message': 'No data records found to display'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

            # Filter dataset columns down to only what is explicitly requested to prevent processing strain
            hover_columns = hover_data if isinstance(hover_data, list) else []
            needed_columns = list(set([xLabel] + hover_columns + value_vars))
            needed_columns = [col for col in needed_columns if col in df.columns]
            df = df[needed_columns]

            # Melt multiple data dimensions into a clean, long-form metrics dataframe 
            df_melted = df.melt(
                id_vars=[xLabel] + hover_columns,
                value_vars=[v for v in value_vars if v in df.columns],
                var_name='Group',
                value_name='Value'
            )
            
            # Safe numerical conversion on value data metric boundaries
            df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce').fillna(0)

            # Assign dynamic chart coordinates based on requested vertical vs horizontal orientation profiles
            is_vertical = orientation.lower() == 'v'
            x_axis_assignment = xLabel if is_vertical else 'Value'
            y_axis_assignment = 'Value' if is_vertical else xLabel

            # Generate the Express multi-variable bar figure
            fig = px.bar(
                df_melted,
                x=x_axis_assignment,
                y=y_axis_assignment,
                color='Group',
                barmode=barmode,
                title=title if title else None,
                hover_data=hover_columns if hover_columns else None,
                orientation=orientation.lower() if orientation.lower() in ['v', 'h'] else 'v'
            )

            # Apply elegant layouts to complement horizontal legend alignments
            fig.update_layout(
                template="plotly_white",
                title_font=dict(size=12),
                title_automargin=True,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(
                    font=dict(size=10),
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=1.02, yanchor="bottom"
                ),
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Build successful response payloads
            message_dict.update({'message': 'Multi column bar chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_flow_chart(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns a flowchart visualization from given process data using Graphviz and NetworkX.

        This method accepts either a data_id to fetch pre-stored data, or a raw data string containing flowchart
        records in JSON format. The resulting flowchart is rendered as a PNG image, base64-encoded, and returned
        as HTML along with meta information and status.

        Edge and node attributes are accessed using `.get()` with sensible defaults, ensuring robust handling of missing fields.
        Edges now support optional custom color and thickness (penwidth).

        Args:
            data_id (str, optional): An identifier used to retrieve stored process data. If provided, overrides `data`.
            data (str, optional): A JSON string with a "records" dictionary, containing "edges" and "nodes" for the flowchart.
            meta_data (dict, optional): Metadata dictionary to include in the response.
            user_id (str, optional): Optional user identifier for logging or auditing purposes.

        Returns:
            dict: A dictionary containing:
                - 'status': 'success' or 'error'
                - 'response': {
                    'meta_data': Meta information and messages,
                    'data': JSON string with an embedded PNG image (base64-encoded HTML figure)
                }
                - 'message': JSON-encoded message describing the outcome.
        """
        try:
            message_dict = {'message': meta_data}
            
            # Use internal string/dict/dataframe parsing router
            validated_response = self._data_validate(data, message_dict)

            if not validated_response.get('status') == 'success':
                message_dict = validated_response.get('message_dict', {})
                message = message_dict.get('message', 'Error passing data')
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
            
            raw_payload = validated_response.get('data')
            
            # Safely navigate nested variations of raw data payloads vs pre-loaded collections
            records = {}
            if isinstance(raw_payload, dict):
                records = raw_payload.get('records', raw_payload)
            elif isinstance(raw_payload, pd.DataFrame):
                # Fallback if raw dataframe rows were built out instead of dict payload structures
                records = {"nodes": raw_payload.to_dict(orient='records'), "edges": []}
            elif isinstance(raw_payload, str):
                try:
                    parsed_json = json.loads(raw_payload)
                    records = parsed_json.get('records', parsed_json)
                except Exception:
                    records = {}

            if not isinstance(records, dict) or (not records.get('nodes') and not records.get('edges')):
                # Fallback step check if the data validate output did not containerize an inner records key
                try:
                    parsed_raw = json.loads(data)
                    records = parsed_raw.get('records', {})
                except Exception:
                    pass

            nodes_list = records.get('nodes', [])
            edges_list = records.get('edges', [])

            if not nodes_list and not edges_list:
                message_dict.update({'message': 'No process structural nodes or edges found to plot'})
                message = json.dumps(message_dict)
                return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

            # Initialize a NetworkX directed chart vector to hold structural positions 
            nx_graph = nx.DiGraph()
            
            # Instantiate the parallel Graphviz rendering pipeline
            dot = graphviz.Digraph(format='png')
            dot.attr(rankdir='TB', size='8,8', dpi='150') # Top-to-Bottom configuration blueprint

            # Track explicit mapped nodes to bound structural edge linkages safely
            tracked_nodes = set()

            # Process block definitions with fallback attribute dictionaries
            for node in nodes_list:
                if not isinstance(node, dict):
                    continue
                label = str(node.get('label', '')).strip()
                if not label:
                    continue
                
                shape = node.get('shape', 'ellipse')
                style = node.get('style', 'filled')
                fillcolor = node.get('fillcolor', '#bbbbbb')
                fontcolor = node.get('fontcolor', 'black')

                nx_graph.add_node(label, shape=shape, style=style, fillcolor=fillcolor, fontcolor=fontcolor)
                dot.node(label, label=label, shape=shape, style=style, fillcolor=fillcolor, fontcolor=fontcolor)
                tracked_nodes.add(label)

            # Process relational path connections
            for edge in edges_list:
                if not isinstance(edge, dict):
                    continue
                start = str(edge.get('start', '')).strip()
                end = str(edge.get('end', '')).strip()
                
                if not start or not end:
                    continue
                
                edge_label = edge.get('label', '')
                color = edge.get('color', 'black')
                penwidth = str(edge.get('penwidth', '1'))

                # Automatically instantiate inferred path endpoints if missing from node declaration objects
                if start not in tracked_nodes:
                    dot.node(start, label=start, shape='ellipse', style='filled', fillcolor='#bbbbbb', fontcolor='black')
                    tracked_nodes.add(start)
                if end not in tracked_nodes:
                    dot.node(end, label=end, shape='ellipse', style='filled', fillcolor='#bbbbbb', fontcolor='black')
                    tracked_nodes.add(end)

                nx_graph.add_edge(start, end, label=edge_label, color=color, penwidth=penwidth)
                
                # Apply attributes directly onto edge mappings
                edge_attrs = {"color": color, "penwidth": penwidth}
                if edge_label:
                    edge_attrs["label"] = edge_label
                dot.edge(start, end, **edge_attrs)

            # Render canvas elements into a transient binary image memory stream
            png_bytes = dot.pipe()
            base64_encoded = base64.b64encode(png_bytes).decode('utf-8')
            
            # Generate target identification flags for localized document objects
            fig_id = f"flow_{str(uuid.uuid4())[:8]}"
            
            # Wrap standard container markup matching responsive structural interfaces
            fig_return = f'<div id="{fig_id}" style="width:100%; overflow:auto; text-align:center;"><img src="data:image/png;base64,{base64_encoded}" style="max-width:100%; height:auto;" /></div>'

            # Build successful response payloads
            message_dict.update({'message': 'Flow chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def plot_flow_chart_plotly(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns an interactive flowchart visualization from process data using Plotly and NetworkX.
        Nodes are automatically organized into structured chronological layers, and paths include dynamic vector arrows.
        """
        message_dict = {'message': meta_data}
        
        # Route through core structural data validator
        validated_response = self._data_validate(data, message_dict)

        if not validated_response.get('status') == 'success':
            message_dict = validated_response.get('message_dict', {})
            message = message_dict.get('message', 'Error passing data')
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}
        
        raw_payload = validated_response.get('data')
        include_plotlyjs = True

        # Safely parse inner records blocks
        records = {}
        if isinstance(raw_payload, dict):
            records = raw_payload.get('records', raw_payload)
        elif isinstance(raw_payload, str):
            try:
                parsed_json = json.loads(raw_payload)
                records = parsed_json.get('records', parsed_json)
            except Exception:
                records = {}

        states = records.get('edges', [])
        node_properties = records.get('nodes', [])

        if not states and not node_properties:
            message_dict.update({'message': 'No data records found to process flow map'})
            message = json.dumps(message_dict)
            return {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        try:
            # Build directed structural matrix 
            G = nx.DiGraph()
            edge_labels = {}
            edge_styles = {}

            # Process links and strip split tags if present
            for state in states:
                if not isinstance(state, dict):
                    continue
                start = str(state.get('start', '')).split(':')[0].strip()
                end = str(state.get('end', '')).split(':')[0].strip()
                
                if not start or not end:
                    continue
                    
                label = state.get('label', '')
                color = state.get('color', '#333333')
                penwidth = float(state.get('penwidth', 2))

                G.add_edge(start, end)
                edge_labels[(start, end)] = label
                edge_styles[(start, end)] = {'color': color, 'width': penwidth}

            # Inject node configuration values
            for node_p in node_properties:
                if not isinstance(node_p, dict):
                    continue
                lbl = str(node_p.get('label', '')).strip()
                if lbl:
                    if lbl not in G.nodes:
                        G.add_node(lbl)
                    G.nodes[lbl].update(node_p)

            # Assign generation layers to nodes to create a structured hierarchy
            # Finds root components and steps downward to calculate hierarchical depth positions
            roots = [n for n, d in G.in_degree() if d == 0]
            for node in G.nodes:
                min_depth = 0
                for root in roots:
                    if nx.has_path(G, root, node):
                        min_depth = max(min_depth, len(nx.shortest_path(G, root, node)) - 1)
                G.nodes[node]['layer'] = min_depth

            # Calculate crisp geometric node positions using a layered multipartite schema
            pos = nx.multipartite_layout(G, subset_key='layer', align='vertical')

            # Build line and directional arrow traces
            edge_traces = []
            arrow_x, arrow_y = [], []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                style = edge_styles.get(edge, {'color': '#333333', 'width': 2})

                # Create separate scatter traces for individual edge colors
                single_edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=style['width'], color=style['color']),
                    hoverinfo='none', mode='lines', showlegend=False
                )
                edge_traces.append(single_edge_trace)

                # Vector logic: Calculate midpoint offsets to inject directional path arrows
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                length = np.sqrt(dx**2 + dy**2) if (dx**2 + dy**2) > 0 else 1
                
                # Scale arrow size parameters relatively
                ux, uy = dx / length, dy / length
                arrow_x += [mx, mx - 0.04 * ux + 0.02 * uy, mx - 0.04 * ux - 0.02 * uy, mx, None]
                arrow_y += [my, my - 0.04 * uy - 0.02 * ux, my - 0.04 * uy + 0.02 * ux, my, None]

            # Vector arrow trace object
            arrow_trace = go.Scatter(
                x=arrow_x, y=arrow_y, fill='toself',
                fillcolor='#333333', line=dict(color='#333333', width=1),
                hoverinfo='none', mode='lines', showlegend=False
            )
            edge_traces.append(arrow_trace)

            # Build structural node metadata positions
            node_x, node_y, node_text, node_color, node_font_color = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_info = G.nodes[node]
                node_text.append(f"<b>{str(node_info.get('label', node))}</b>")
                node_color.append(node_info.get('fillcolor', '#E5ECF6'))
                node_font_color.append(node_info.get('fontcolor', '#000000'))

            # Dynamic node trace object
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(
                    size=45, color=node_color, 
                    line=dict(width=2, color='#1A1A1A'), 
                    shape='square'
                ),
                text=node_text, textposition="middle center",
                textfont=dict(color=node_font_color, size=11),
                hoverinfo='text', showlegend=False
            )

            # Build edge annotation text labels
            edge_label_x, edge_label_y, edge_label_text = [], [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                lbl = edge_labels.get(edge, "")
                if lbl:
                    # Place label text at 1/3 position of path vector to clear center arrows
                    edge_label_x.append(x0 + (x1 - x0) * 0.35)
                    edge_label_y.append(y0 + (y1 - y0) * 0.35)
                    edge_label_text.append(f"<span style='background-color:white; px-1;'>{lbl}</span>")

            edge_label_trace = go.Scatter(
                x=edge_label_x, y=edge_label_y, mode='text',
                text=edge_label_text, textposition='top center',
                textfont=dict(size=9, color='#555555'),
                hoverinfo='none', showlegend=False
            )

            # Compile layout graph
            fig = go.Figure(
                data=[*edge_traces, node_trace, edge_label_trace],
                layout=go.Layout(
                    hovermode='closest',
                    margin=dict(b=40, l=40, r=40, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constraints_allowed='domain'),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )

            plot_html = pio.to_html(
                fig,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
                include_plotlyjs=include_plotlyjs
            )
            
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace('<div>', f'<div id="{fig_id}">')

            # Package output payloads
            message_dict.update({'message': 'Flow chart plotted'})
            message = json.dumps(message_dict)
            response = {'status': 'success', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': fig_return}), 'message': message}}

        except Exception as e:
            message_dict.update({'message': f'Error: {str(e)}'})
            message = json.dumps(message_dict)
            response = {'status': 'error', 'response': {'meta_data': message_dict, 'data': json.dumps({'figure': ''})}, 'message': message}

        return response
    def display_image(self, result):
        """
        Renders the generated Plotly figure or Flowchart HTML string directly 
        inside notebook outputs (e.g., Google Colab, Jupyter Notebook).

        Args:
            result (dict): The output dictionary returned by any of the plotting methods.
        """
        if not isinstance(result, dict):
            print("Failed to plot: Invalid result format provided.")
            return

        if result.get('status') == 'success':
            try:
                # Safely parse the inner response data packet
                response_payload = result.get('response', {})
                data_str = response_payload.get('data', '{}')
                
                # Handle cases where data is already a dict vs a serialized JSON string
                if isinstance(data_str, str):
                    response_data = json.loads(data_str)
                else:
                    response_data = data_str

                plot_html = response_data.get('figure', '')

                if plot_html:
                    # Render the HTML block directly inside the notebook canvas
                    display(HTML(plot_html))
                else:
                    print("Failed to plot: No figure matrix data found in response payload.")
                    
            except Exception as e:
                print(f"Failed to render visualization component: {str(e)}")
        else:
            # Safely extract error metrics across your layout's payload layers
            response_payload = result.get('response', {})
            error_msg = response_payload.get('message') or result.get('message') or "Unknown processing error"
            print(f"Failed to plot: {error_msg}")
 
    