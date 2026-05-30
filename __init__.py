"""
data_wrangling
==============
A robust Python toolkit for data wrangling, feature engineering,
interactive visualisation, and statistical association analysis —
optimised for Google Colab.

Quick start
-----------
>>> from data_wrangling import DataWrangler, ChartBuilder
>>> wrangler = DataWrangler()
>>> wrangler.load_url("https://example.com/data.csv")
>>> wrangler.get_profile()
>>> wrangler.fix_nulls(strategy="median")
>>> wrangler.plot_numeric()
"""

from .core import DataWrangler, ChartBuilder

__all__ = ["DataWrangler", "ChartBuilder"]
__version__ = "1.0.0"
__author__ = "DataWranglerTool"
