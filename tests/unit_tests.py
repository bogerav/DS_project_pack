import os.path
import pandas as pd
import numpy as np

from dspack.data import generate_data
from dspack.plotting import plot_analysis
from dspack.analysis import analyse_data

TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", 'tests', 'test_data')
)

def test_data_generator() -> None:
    start = pd.read_parquet(os.path.join(TEST_DATA_DIR, "raw_data.parquet"))
    generated = generate_data()
    assert start.equals(generated)

def test_analyse_data() -> None:
    start_fit = pd.read_parquet(os.path.join(TEST_DATA_DIR, "fit_results.parquet"))
    start = pd.read_parquet(os.path.join(TEST_DATA_DIR, "raw_data.parquet"))
    new_fit = analyse_data(start)
    assert start_fit.equals(new_fit)

def test_full_analysis() -> None:
    raw_data = generate_data()
    fit_results = analyse_data(raw_data)
    plot_analysis(raw_data = raw_data,
                  fit_results = fit_results)
