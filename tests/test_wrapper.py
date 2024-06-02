import numpy as np
import os
from pathlib import Path
import pytest
import xarray as xr

import sys

repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
data_dir = repo_root / "data"
sys.path.append(str(src_dir))

from analysis import GEM, waccmwind


def test_gem_initialization_no_errors():
    """
    Test the initialization of the GEM object without any errors.
    """
    gem = GEM()

    # Basic attribute existence checks
    assert hasattr(gem, "solar_farms")
    assert hasattr(gem, "wind_farms")
    assert hasattr(gem, "solar_maxyear")
    assert hasattr(gem, "wind_maxyear")


def test_get_solar_flux_time_series():
    """
    Test function for the 'get_solar_flux_time_series' function.
    """
    lat, lon = 30.0, -90.0
    time, series, baseline = GEM.get_solar_flux_time_series(lat, lon)

    assert len(time) == 180  # 15 years * 12 months
    assert len(series) == 180
    assert len(baseline) == 12
    assert np.isclose(baseline.sum(), 1.0)  # Check normalization
    assert np.all(series >= 0)  # Values should be non-negative


def test_get_wind_power_time_series():
    """
    Test function for the `get_wind_power_time_series` function in the GEM module.
    """
    lat, lon = 45.0, 15.0
    time, series, baseline = GEM.get_wind_power_time_series(lat, lon)

    assert len(time) == 156  # 13 years * 12 months
    assert len(series) == 156
    assert len(baseline) == 12
    assert np.isclose(baseline.sum(), 1.0)  # Check normalization
    assert np.all(series >= 0)  # Values should be non-negative
    assert -180 <= lon <= 180  # Check if longitude is wrapped between -180 and 180


# Test Data
TEST_DATA_DIR = f"{data_dir}/wind-data"  # Replace with your actual test data directory
TEST_YEARS = [1, 5]  # Example years to test
TEST_MONTHS = [1, 6, 12]  # Example months to test
TEST_SIMS = ["control", "catastrophe"]


# Parameterized Test
@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("month", TEST_MONTHS)
@pytest.mark.parametrize("sim", TEST_SIMS)
def test_waccmwind_get(year, month, sim):
    # Construct the expected file path
    expected_filename = f"{TEST_DATA_DIR}/windspeed_{sim}_{year + 4:02}.nc"

    # Check if the file exists
    assert os.path.exists(
        expected_filename
    ), f"Test data file not found: {expected_filename}"

    # Call the `get` method
    ds = waccmwind.get(year, month, sim)

    # Assertions
    assert isinstance(ds, xr.Dataset), "The returned object should be an xarray Dataset"
    assert "windspeed" in ds, "The Dataset should contain the 'windspeed' variable"
