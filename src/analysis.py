from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr


plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class GEM:
    def __init__(self):
        self.solar_farms = self.load_solar_farms()
        self.wind_farms = self.load_wind_farms()

    def load_solar_farms(self):
        """
        Load solar farm data from Global Energy Monitor.

        Only projects that are operating, announced, in construction, or in pre-construction are included.

        Returns:
            pd.DataFrame: solar farm data
        """
        file_path = "../data/global-energy-monitor/Global-Solar-Power-Tracker-December-2023.xlsx"
        largescale = pd.read_excel(file_path, sheet_name=1)
        mediumscale = pd.read_excel(file_path, sheet_name=2)
        smallscale = pd.read_excel(file_path, sheet_name=3)
        combined_df = pd.concat(
            [largescale, mediumscale, smallscale], ignore_index=True
        )
        # drop cancelled projects
        combined_df = combined_df[combined_df["Status"] != "cancelled"]
        # drop shelved projects
        combined_df = combined_df[combined_df["Status"] != "shelved"]
        # drop retired projects
        combined_df = combined_df[combined_df["Status"] != "retired"]
        # drop mothballed projects
        combined_df = combined_df[combined_df["Status"] != "mothballed"]
        return combined_df[
            ["Country", "Latitude", "Longitude", "Capacity (MW)", "Status"]
        ]

    def load_wind_farms(self):
        """
        Load wind farm data from Global Energy Monitor.

        Only projects that are operating, announced, in construction, or in pre-construction are included.

        Returns:
            pd.DataFrame: wind farm data
        """
        file_path = (
            "../data/global-energy-monitor/Global-Wind-Power-Tracker-December-2023.xlsx"
        )
        largescale = pd.read_excel(file_path, sheet_name=1)
        smallscale = pd.read_excel(file_path, sheet_name=2)
        combined_df = pd.concat([largescale, smallscale], ignore_index=True)
        # drop cancelled projects
        combined_df = combined_df[combined_df["Status"] != "cancelled"]
        # drop shelved projects
        combined_df = combined_df[combined_df["Status"] != "shelved"]
        # drop retired projects
        combined_df = combined_df[combined_df["Status"] != "retired"]
        # drop mothballed projects
        combined_df = combined_df[combined_df["Status"] != "mothballed"]
        return combined_df[
            ["Country", "Latitude", "Longitude", "Capacity (MW)", "Status"]
        ]

    def sum_operating_solar_farms_per_country(self):
        """
        Returns the total power operating capacity for the top 5 countries with solar farms.

        Useful for validation.

        Returns:
            pd.Series: total power operating capacity for the top 5 countries with solar farms
        """
        return (
            self.solar_farms[self.solar_farms.Status == "operating"]
            .groupby("Country")["Capacity (MW)"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

    def sum_operating_wind_farms_per_country(self):
        """
        Returns the total power operating capacity for the top 5 countries with wind farms.

        Useful for validation.

        Returns:
            pd.Series: total power operating capacity for the top 5 countries with wind farms
        """
        return (
            self.wind_farms[self.wind_farms.Status == "operating"]
            .groupby("Country")["Capacity (MW)"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

    def plot_solar_farm_map(self):
        """
        Plot the location of solar farms on a map.
        """
        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)
        # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax.scatter(
            self.solar_farms.Longitude,
            self.solar_farms.Latitude,
            s=0.01 * self.solar_farms["Capacity (MW)"],
            color="orange",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        plt.show()

    def plot_wind_farm_map(self):
        """
        Plot the location of wind farms on a map.
        """
        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.5)
        # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax.scatter(
            self.wind_farms.Longitude,
            self.wind_farms.Latitude,
            s=0.01 * self.wind_farms["Capacity (MW)"],
            color="blue",
            alpha=0.2,
            transform=ccrs.PlateCarree(),
        )
        plt.show()

    @staticmethod
    def get_solar_flux_time_series(lat, lon):
        """
        Get the time series of solar flux at a given location. The solar flux is normalized by the
        baseline solar flux at the same location, so that 1 is the baseline and 0 is no solar flux.
        The baseline is the average solar flux for each month for the last year of the simulation
        (year 16).

        Args:
            lat (float): latitude
            lon (float): longitude

        Returns:
            np.array: time
            np.array: series of solar flux (1 is the baseline, 0 is no solar flux)
            np.array: seasonal variation of solar flux (baseline)
        """
        year_base = 16
        var = "FSDS"
        time = []
        series = []
        baseline_values = {}
        for month in range(1, 13):
            ref_data = waccm.get(var, year_base, month).isel(time=0)
            ref_data_value = ref_data.sel(lat=lat, lon=lon, method="nearest")
            baseline_values[month] = ref_data_value
        for year in range(1, 16):
            for month in range(1, 13):
                time.append(waccm.months_since_nw(year, month))
                data = waccm.get(var, year, month).isel(time=0)
                data_value = data.sel(lat=lat, lon=lon, method="nearest")
                ref_data_value = baseline_values[month]
                series.append(data_value / ref_data_value)
        time = np.array(time)
        series = np.array(series)
        baseline = np.array(list(baseline_values.values()))
        baseline = baseline / baseline.sum()
        return time, series, baseline

    @staticmethod
    def wind_power_output(v, v_cut_in=3, v_rated=15, v_cut_out=25, P_rated=1):
        """
        Calculate the wind turbine power output given a wind speed.

        Arges:
            v (float): Wind speed in meters per second (m/s).
            v_cut_in (float): Cut-in wind speed (m/s). Default is 3 m/s.
            v_rated (float): Rated wind speed (m/s). Default is 15 m/s.
            v_cut_out (float): Cut-out wind speed (m/s). Default is 25 m/s.
            P_rated (float): Max power.

        Returns:
            float: Power output.
        """
        if v < v_cut_in or v > v_cut_out:
            return 0
        elif v_cut_in <= v < v_rated:
            return P_rated * ((v - v_cut_in) / (v_rated - v_cut_in)) ** 3
        elif v_rated <= v <= v_cut_out:
            return P_rated
        else:
            return 0

    @staticmethod
    def get_power_time_series(lat, lon, power_type):
        """
        Get the time series of wind or solar power output at a given location. The power is normalized by the
        baseline power at the same location, so that 1 is the baseline and 0 is no power.

        Args:
            lat (float): latitude
            lon (float): longitude
            power_type (str): type of power ("wind" or "solar")

        Returns:
            np.array: time
            np.array: series of power (1 is the baseline, 0 is no power)
            np.array: seasonal variation of power (baseline)
        """
        if power_type == "wind":
            var = "windspeed"
        elif power_type == "solar":
            var = "a2x3h_Faxa_swndr"
        time = []
        series = []
        baseline_values = {}
        lon = (lon + 360) % 360
        for month in range(1, 13):
            year_base = 5
            ref_data = waccmdaily.get(var, f"{year_base:02}-{month:02}", sim="control")
            ref_data_value = ref_data.sel(a2x3h_ny=lat, a2x3h_nx=lon, method="nearest")
            if power_type == "wind":
                ref_data_value = GEM.wind_power_output(ref_data_value)
            baseline_values[month] = ref_data_value
        for year in range(1, 11):
            for month in range(1, 13):
                time.append(waccm.months_since_nw(year, month))
                data = waccmdaily.get(var, f"{year:02}-{month:02}", sim="NW")
                data_value = data.sel(a2x3h_ny=lat, a2x3h_nx=lon, method="nearest")
                if power_type == "wind":
                    data_value = GEM.wind_power_output(data_value)
                series.append(data_value / baseline_values[month])
        time = np.array(time)
        series = np.array(series)
        baseline = np.array(list(baseline_values.values()))
        baseline = baseline / baseline.sum()
        return time, series, baseline

    def get_solar_flux_for_farm(self, row):
        """
        Helper function to compute the solar flux time series for a single solar farm.
        This function will be called in parallel for each solar farm.
        """
        lat = row["Latitude"]
        lon = row["Longitude"]
        capacity = row["Capacity (MW)"]
        time, series, baseline = self.get_solar_flux_time_series(lat, lon)
        weighted_series = series * capacity
        weighted_baseline = baseline * capacity
        return weighted_series, weighted_baseline

    def get_country_solar_power_time_series(self, country):
        """
        Average solar power variation compared to baseline for a given country.
        We are averaging over the locations of all solar farms in the country,
        weighted by their capacity.

        Args:
            country (str): country name

        Returns:
            np.array: time
            np.array: country-aggregated series of solar power variation compared to baseline
                (1 is the baseline, 0 is no solar power)
        """
        df = self.solar_farms[self.solar_farms.Country == country]
        total_capacity = df["Capacity (MW)"].sum()
        country_aggregated_series = np.zeros(180)
        country_aggregated_baseline = np.zeros(12)

        pbar = tqdm.tqdm(total=len(df))

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_solar_flux_for_farm, row): row["Capacity (MW)"]
                for _, row in df.iterrows()
            }

            for future in as_completed(futures):
                weighted_series, weighted_baseline = future.result()
                country_aggregated_series += weighted_series / total_capacity
                country_aggregated_baseline += weighted_baseline / total_capacity
                pbar.update(1)
        pbar.close()

        time, _, _ = self.get_solar_flux_time_series(0, 0)
        return time, country_aggregated_series, country_aggregated_baseline

    def get_all_country_solar_power_time_series(
        self,
        output_csv1="../results/fraction_of_solar_power_countries.csv",
        output_csv2="../results/baseline_seasonality_solar_power_countries.csv",
    ):
        """
        Calculates the average solar power variation compared to baseline for all countries.

        Output is saved to a CSV file.

        Args:
            output_csv1 (str): path to the output CSV file for the solar power variation
            output_csv2 (str): path to the output CSV file for the baseline seasonality
        """
        # Sort countries alphabetically
        countries = sorted(self.solar_farms["Country"].unique())

        # Check if the output file already exists to determine where to resume
        try:
            existing_df = pd.read_csv(output_csv1)
            existing_df_baseline = pd.read_csv(output_csv2)
            completed_countries = existing_df.columns[1:]  # Exclude 'Time' column
            countries_to_process = [
                c for c in countries if c not in completed_countries
            ]
        except FileNotFoundError:
            # If file does not exist, start from scratch
            existing_df = None
            existing_df_baseline = None
            countries_to_process = countries

        # Process each country
        for country in countries_to_process:
            print(f"Processing {country}...")
            time, series, baseline = self.get_country_solar_power_time_series(country)

            # If this is the first country being processed, initialize DataFrame
            if existing_df is None:
                existing_df = pd.DataFrame(time, columns=["Months_after_NW"])
                existing_df[country] = series
            else:
                existing_df[country] = series

            # same with baseline
            if existing_df_baseline is None:
                existing_df_baseline = pd.DataFrame(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    columns=["Month_of_the_year"],
                )
                existing_df_baseline[country] = baseline
            else:
                existing_df_baseline[country] = baseline

            # Write (or overwrite) the CSV file with updated data
            existing_df.to_csv(output_csv1, index=False)
            print(f"Saved {country} to {output_csv1}")

            existing_df_baseline.to_csv(output_csv2, index=False)
            print(f"Saved {country} to {output_csv2}")

    def postprocess_aggregate_countries_solar(self, input_csv):
        """
        Calculate a weighted mean of the solar power variation for all countries.
        Weights are given by the total solar power capacity of each country.

        Args:
            input_csv (str): path to the input CSV file, created by get_all_country_solar_power_time_series

        Returns:
            None, but displays a plot
        """
        df = pd.read_csv(input_csv)
        df = df.set_index("Months_after_NW")
        # iterate over columns with a loop
        for col in df.columns:
            df[col] = (
                df[col]
                * self.solar_farms[self.solar_farms.Country == col][
                    "Capacity (MW)"
                ].sum()
            )
        weighted_mean = (
            df.iloc[:, :].sum(axis=1) / self.solar_farms["Capacity (MW)"].sum()
        )

        # make figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index / 12, weighted_mean)
        ax.set_xlabel("Years after nuclear war")
        ax.set_ylabel("Solar power compared to baseline")
        plt.show()
        return

    def postprocess_solar_map(
        self, fraction_csv_file, baseline_csv_file, zmin=None, zmax=None
    ):
        """
        Makes a map of solar power reduction over the year in each country.

        Seasonality is taken into consideration when calculating the reduction over the year.

        Args:
            fraction_csv_file (str): path to the CSV file with the fraction of solar power
            baseline_csv_file (str): path to the CSV file with the baseline seasonality
            zmin (float): minimum value for the color scale
            zmax (float): maximum value for the color scale

        Returns:
            None, but displays a plot
        """
        for year in range(1, 15):
            fraction_csv = pd.read_csv(fraction_csv_file)
            fraction_csv = fraction_csv.loc[year * 12 : (year + 1) * 12 - 1]
            baseline_csv = pd.read_csv(baseline_csv_file)

            country_dict = {}

            # loop over countries:
            for col in fraction_csv.columns:
                if col != "Months_after_NW":
                    fraction = fraction_csv[col].to_numpy()
                    baseline = baseline_csv[col].to_numpy()
                    reduction = np.dot(fraction, baseline)
                    country_dict[col] = reduction * 100

            name_mapping = {
                "United States": "United States of America",
                "DR Congo": "Dem. Rep. Congo",
                "Republic of the Congo": "Congo",
                "Dominican Republic": "Dominican Rep.",
                "Türkiye": "Turkey",
                "South Sudan": "S. Sudan",
                "Central African Republic": "Central African Rep.",
                "Czech Republic": "Czechia",
                "Bosnia and Herzegovina": "Bosnia and Herz.",
            }
            country_dict = {name_mapping.get(k, k): v for k, v in country_dict.items()}

            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world["solar_reduction"] = world["name"].map(country_dict)
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            world.boundary.plot(ax=ax, linewidth=0.5, color="k")
            world.dropna(subset=["solar_reduction"]).plot(
                column="solar_reduction",
                ax=ax,
                legend=True,
                legend_kwds={
                    "label": "Solar power compared to baseline (%)",
                    "orientation": "vertical",
                    "shrink": 0.5,
                },
                cmap="viridis",
                vmin=zmin,
                vmax=zmax,
            )
            world[world["solar_reduction"].isna()].plot(ax=ax, color="lightgrey")

            ax.set_axis_off()
            ax.grid(False)
            plt.title(f"Year {year} after nuclear war")

            plt.savefig(f"../results/solar_reduction_map_{year:02}.pdf", dpi=300)

        os.system(
            "pdfunite ../results/solar_reduction_map_*.pdf ../results/solar_reduction_map.pdf"
        )
        os.system("rm ../results/solar_reduction_map_*.pdf")


class waccm:
    """
    Used to access, process, and visualize simulation data produced by the Coupe et al. 2019
    nuclear winter simulation using the Whole Atmosphere Community Climate Model (WACCM).

    This is for the publicly available dataset, which has a monthly resolution and does not include
    wind data.
    """

    @staticmethod
    def get(var, year, month, use_dask=False):
        """
        Get data for a given variable, year, and month.

        Year 1 is the year of the nuclear war (which starts in May).

        Args:
            var (str): variable name
            year (int): year
            month (int): month
            use_dask (bool): whether to use Dask for chunking

        Returns:
            xr.DataArray: data
        """
        year = year + 4
        year = str(year).zfill(4)
        month = str(month).zfill(2)
        ds = xr.open_dataset(
            f"../data/nw_ur_150_07_mini/nw_ur_150_07.cam.h0.{year}-{month}.nc",
            chunks={} if use_dask else None,  # Use Dask for chunking if requested
        )
        return ds[var]

    @staticmethod
    def plot_map(var, year, month, lev=None):
        """
        Plot data for a given variable, year, and month on a map.

        Args:
            var (str): variable name
            year (int): year
            month (int): month
            lev (float): level to plot (hPa)
        """
        toplot = waccm.get(var, year, month)
        if lev is not None:
            # find closest level
            lev = toplot.lev.sel(lev=lev, method="nearest")
            print(f"Level: {lev.values} hPa")
            toplot = toplot.sel(lev=lev)
        toplot_cyclic, lon_cyclic = add_cyclic_point(
            toplot.isel(time=0), coord=toplot.lon
        )
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        cbar = "viridis"
        if var == "CRSOOTMR":
            cbar = "pink_r"
        contour = ax.contourf(
            lon_cyclic,
            toplot.lat,
            toplot_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cbar,
            levels=16,
        )
        plt.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            label=waccm.label(var),
        )
        plt.title(
            f"{waccm.month_name(month)} of Year {year} ({waccm.months_since_nw(year, month)} months since nuclear war)"
        )
        plt.tight_layout()

    @staticmethod
    def plot_map_diff(var, year, month, year_base=16, relative=False, lev=None):
        """
        Plot difference in data for a given variable, year, and month on a map.

        Args:
            var (str): variable name
            year (int): year
            month (int): month
            year_base (int): base year for comparison, the default is 16, which
                corresponds to the end year of the simulation when things more or
                less return to normal
            relative (bool): whether to plot relative difference
            lev (float): level to plot (hPa)
        """
        month_base = month
        data_current = waccm.get(var, year, month).isel(time=0).load()
        data_base = waccm.get(var, year_base, month_base).isel(time=0).load()
        if lev is not None:
            # find closest level
            lev = data_current.lev.sel(lev=lev, method="nearest")
            print(f"Level: {lev.values} hPa")
            data_current = data_current.sel(lev=lev)
            data_base = data_base.sel(lev=lev)
        if relative:
            toplot = 100 * (data_current - data_base) / data_base
            cmap = "PuOr_r"
        else:
            toplot = data_current - data_base
            cmap = "viridis"
        toplot_cyclic, lon_cyclic = add_cyclic_point(toplot, coord=toplot.lon)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        if relative:
            levels = np.linspace(-100, 100, 21)
        else:
            levels = 15
        contour = ax.contourf(
            lon_cyclic,
            toplot.lat,
            toplot_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            levels=levels,
        )
        plt.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            label=waccm.label(var, relative=relative),
        )
        plt.title(
            f"{waccm.month_name(month)} of Year {year} ({waccm.months_since_nw(year, month)} months since nuclear war)"
        )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_series_global_average(
        var, year_base=16, lev=None, diff=False, relative=False
    ):
        """
        Averages var over the globe and plots the difference from the base year
        for each month.

        Args:
            var (str): variable name
            year_base (int): base year for comparison, the default is 16, which
                corresponds to the end year of the simulation when things more or
                less return to normal
            lev (float): level to plot (hPa)
            diff (bool): whether to plot difference wrt base year
            relative (bool): whether to plot relative difference
        """
        if not diff and relative:
            raise ValueError("Cannot plot relative difference without diff=True")
        time = []
        series = []
        for year in range(1, 16):
            for month in range(1, 13):
                time.append(waccm.months_since_nw(year, month))
                data = waccm.get(var, year, month).isel(time=0)
                ref_data = waccm.get(var, year_base, month).isel(time=0)
                data_global_average = waccm.global_average(data, lev=lev)
                ref_data_global_average = waccm.global_average(ref_data, lev=lev)
                if diff and relative:
                    series.append(
                        100
                        * (data_global_average - ref_data_global_average)
                        / ref_data_global_average
                    )
                elif diff:
                    series.append(data_global_average - ref_data_global_average)
                else:
                    series.append(data_global_average)
        time = np.array(time)
        plt.plot(time / 12, series)
        plt.xlabel("Years since nuclear war")
        plt.ylabel(waccm.label(var, relative=relative))
        plt.tight_layout()
        plt.title("Global average")

    @staticmethod
    def plot_time_series_location(
        var, lat, lon, year_base=16, lev=None, diff=False, relative=False
    ):
        """
        Plots the difference from the base year for each month for a given location.

        Args:
            var (str): variable name
            year_base (int): base year for comparison, the default is 16, which
                corresponds to the end year of the simulation when things more or
                less return to normal
            lat (list): latitudes of locations to plot
            lon (list): longitudes of locations to plot
            lev (float): level to plot (hPa)
            diff (bool): whether to plot difference wrt base year
            relative (bool): whether to plot relative difference
        """
        if not diff and relative:
            raise ValueError("Cannot plot relative difference without diff=True")
        lat = np.array(lat)
        lon = np.array(lon)
        ls_list = ["-", "--", "-.", ":"] * 10
        for lat_val, lon_val in zip(lat, lon):
            time = []
            series = []
            for year in range(1, 16):
                for month in range(1, 13):
                    time.append(waccm.months_since_nw(year, month))
                    data = waccm.get(var, year, month).isel(time=0)
                    ref_data = waccm.get(var, year_base, month).isel(time=0)
                    data_value = data.sel(lat=lat_val, lon=lon_val, method="nearest")
                    ref_data_value = ref_data.sel(
                        lat=lat_val, lon=lon_val, method="nearest"
                    )
                    if diff and relative:
                        series.append(
                            100 * (data_value - ref_data_value) / ref_data_value
                        )
                    elif diff:
                        series.append(data_value - ref_data_value)
                    else:
                        series.append(data_value)
            time = np.array(time)
            ls = ls_list.pop(0)
            plt.plot(
                time / 12, series, label=waccm.format_coords(lat_val, lon_val), ls=ls
            )
        plt.xlabel("Years since nuclear war")
        plt.ylabel(waccm.label(var, relative=relative))
        plt.tight_layout()
        plt.legend()
        plt.show()

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        for lat_val, lon_val in zip(lat, lon):
            ax.plot(lon_val, lat_val, "o", markersize=5, transform=ccrs.Geodetic())
        plt.show()

    @staticmethod
    def label(var, relative=False):
        """
        Get a label for a given variable.

        Args:
            var (str): variable name
            relative (bool): whether to plot relative difference

        Returns:
            str: label
        """
        if var == "FSDS":
            return (
                "Change in downwelling solar flux at surface (%)"
                if relative
                else "Downwelling solar flux at surface (W/m²)"
            )
        elif var == "FSNS":
            return (
                "Change in net solar flux at surface (%)"
                if relative
                else "Net solar flux at surface (W/m²)"
            )
        elif var == "PRECC":
            return (
                "Change in convective precipitation rate (%)"
                if relative
                else "Convective precipitation rate (m/s)"
            )
        elif var == "PRECL":
            return (
                "Change in large-scale precipitation rate (%)"
                if relative
                else "Large-scale precipitation rate (m/s)"
            )
        elif var == "PS":
            return (
                "Change in surface pressure (%)"
                if relative
                else "Surface pressure (Pa)"
            )
        elif var == "Q":
            return (
                "Change in specific humidity (%)"
                if relative
                else "Specific humidity (kg/kg)"
            )
        elif var == "RELHUM":
            return (
                "Change in relative humidity (%)"
                if relative
                else "Relative humidity (%)"
            )
        elif var == "T":
            return "Change in temperature (%)" if relative else "Temperature (K)"
        elif var == "Z3":
            return (
                "Change in geopotential height (%)"
                if relative
                else "Geopotential height above sea level (m)"
            )
        elif var == "AEROD_v":
            return (
                "Change in aerosol optical depth (%)"
                if relative
                else "Aerosol optical depth (visible band)"
            )
        elif var == "CRSOOTMR":
            return (
                "Change in soot mass mixing ratio (%)"
                if relative
                else "Soot mass mixing ratio (kg/kg)"
            )
        elif var == "CRSOOTRE":
            return (
                "Change in soot effective radius (%)"
                if relative
                else "Soot effective radius (um)"
            )
        else:
            return var

    @staticmethod
    def month_name(month_number):
        """
        Get a month name for a given month number.

        Args:
            month_number (int): month number

        Returns:
            str: month name
        """
        return [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "Decemeber",
        ][month_number - 1]

    @staticmethod
    def months_since_nw(year, month):
        """
        Calculate the number of months since the nuclear war.

        Args:
            year (int): year
            month (int): month

        Returns:
            int: number of months since nuclear war
        """
        return (year - 1) * 12 + month - 5

    @staticmethod
    def global_average(da, lev=None):
        """
        Calculate the global average of an xarray DataArray, weighted by the surface area of each cell.

        Args:
            da (xarray.DataArray): DataArray containing the variable to be averaged.
            lev (float): level to average (hPa)

        Returns:
            float: Global average of the variable.
        """
        if lev is not None:
            # find closest level
            lev = da.lev.sel(lev=lev, method="nearest")
            print(f"Level: {lev.values} hPa")
            da = da.sel(lev=lev)

        # Earth's radius in meters
        R = 6371000

        if "lat" not in da.coords or "lon" not in da.coords:
            raise ValueError("DataArray must have 'lat' and 'lon' coordinates")

        lat_rad = np.deg2rad(da["lat"].values)
        lon_rad = np.deg2rad(da["lon"].values)

        lon_diff = np.diff(lon_rad)
        lon_diff = np.append(lon_diff, lon_diff[0])

        lat_diff = np.diff(np.sin(lat_rad))
        # Append the difference between the last and first (should be zero as they are poles)
        lat_diff = np.append(lat_diff, 0)

        # Calculate the area of each grid cell
        cell_area = R**2 * np.outer(lat_diff, lon_diff)

        # Ensure the cell area has the same dimensions as the DataArray
        cell_area = xr.DataArray(
            cell_area, coords={"lat": da["lat"], "lon": da["lon"]}, dims=["lat", "lon"]
        )

        # Calculate the weighted average
        weighted_sum = (da * cell_area).sum(dim=["lat", "lon"])
        total_area = cell_area.sum(dim=["lat", "lon"])

        answer = weighted_sum / total_area

        # check/assert that that the total area is close to the surface area of the Earth
        assert np.isclose(total_area, 4 * np.pi * R**2, rtol=1e-3)

        return answer.values

    @staticmethod
    def format_coords(lat, lon):
        """
        Format latitude and longitude coordinates to a string with N/S and E/W.

        Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

        Returns:
        str: Formatted coordinates.
        """

        # Round the coordinates to the nearest degree
        lat = round(lat)
        lon = round(lon)

        # Determine the hemisphere indicators
        lat_hemi = "N" if lat >= 0 else "S"
        lon_hemi = "E" if lon >= 0 else "W"

        # Format the string with hemisphere indicators
        formatted_lat = f"{abs(lat)}°{lat_hemi}"
        formatted_lon = f"{abs(lon)}°{lon_hemi}"

        return f"{formatted_lat}, {formatted_lon}"


class waccmdaily:
    """
    Used to access, process, and visualize simulation data produced by the Coupe et al. 2019
    nuclear winter simulation using the Whole Atmosphere Community Climate Model (WACCM).

    This is for a version of the dataset that is not publicly available, which has a daily
    resolution and includes wind data.
    """

    wind_speed_cache = {}

    @staticmethod
    def calculate_and_cache_wind_speed(year, ds):
        """Calculate and cache the wind speed for a given year if not already cached."""
        cache_key = f"windspeed_{year}"
        if cache_key not in waccmdaily.wind_speed_cache:
            u = ds["a2x3h_Sa_u"]
            v = ds["a2x3h_Sa_v"]
            wind_speed = np.sqrt(u**2 + v**2)
            waccmdaily.wind_speed_cache[cache_key] = wind_speed
        return waccmdaily.wind_speed_cache[cache_key]

    @staticmethod
    def get(variable, time, sim):
        """
        Retrieve data for a specific variable at a given time from an xarray dataset,
        loading the dataset from a file based on the year in the time argument.

        Args:
            variable (str): the name of the variable to retrieve
            time (str): the time specification, which can be a specific time, a specific day, or a specific month
                nuclear winter starts in May of year 1 (which is actually year 5 in the dataset)
            sim (str): the simulation to retrieve the data from, either "control" or "NW"

        Returns:
            xarray.DataArray or None: The requested data or None if the operation cannot be completed
        """
        if not waccmdaily.verify_time_format(time):
            print(
                f"Invalid time format: {time}. Expected formats are 'YY-MM-DDTHH:MM:SS', 'YY-MM-DD', or 'YY-MM'."
            )
            return None

        # adjust year by adding 4 to YY in time
        original_year = int(time[:2])
        year = original_year + 4
        year = str(year).zfill(2)
        full_time_str = (
            f"{year}{time[2:]}"  # Reconstruct the full time string with adjusted year
        )
        file_path = os.path.join("..", "data", "daily", f"year{year}.nc")

        try:
            ds = xr.open_dataset(file_path)

            # For windspeed, check if it's already calculated and cached for the year
            if variable == "windspeed":
                ds["windspeed"] = waccmdaily.calculate_and_cache_wind_speed(year, ds)

            full_time_str = f"00{full_time_str}"
            if len(time) == 8:  # Specific day
                data = ds[variable].sel(time=full_time_str).mean(dim="time")
            elif len(time) == 5:  # Specific month
                data = (
                    ds[variable]
                    .sel(time=full_time_str)
                    .resample(time="ME")
                    .mean()
                    .isel(time=0)
                )
            elif len(time) > 8:  # Specific time
                data = ds[variable].sel(time=full_time_str).isel(time=0)

            latitude = np.linspace(-90, 90, data.shape[0])
            longitude = np.linspace(0, 360, data.shape[1], endpoint=False)
            data = data.assign_coords(a2x3h_ny=("a2x3h_ny", latitude), a2x3h_nx=("a2x3h_nx", longitude))

            return data

        except KeyError:
            print(f"Variable '{variable}' not found in the dataset.")
            return None
        except ValueError as e:
            print(f"Error processing time '{time}': {e}")
            return None
        finally:
            if "ds" in locals():
                ds.close()

    @staticmethod
    def verify_time_format(time_str):
        """
        Verify the format of the input time string.

        Args:
            time_str (str): The time string to be verified.

        Returns:
            bool: True if the time string matches any of the allowed formats, False otherwise.

        Expected formats are 'YY-MM-DDTHH:MM:SS', 'YY-MM-DD', or 'YY-MM'.
        """
        specific_time_pattern = r"^\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        specific_day_pattern = r"^\d{2}-\d{2}-\d{2}$"
        specific_month_pattern = r"^\d{2}-\d{2}$"

        if (
            re.match(specific_time_pattern, time_str)
            or re.match(specific_day_pattern, time_str)
            or re.match(specific_month_pattern, time_str)
        ):
            return True
        else:
            return False

    @staticmethod
    def label(var):
        """
        Get a label for a given variable.

        Args:
            var (str): variable name

        Returns:
            str: label
        """
        if var == "a2x3h_Sa_u":
            return "Eastward wind (m/s)"
        elif var == "a2x3h_Sa_v":
            return "Northward wind (m/s)"
        elif var == "a2x3h_Sa_tbot":
            return "Surface temperature (K)"
        elif var == "a2x3h_Sa_shum":
            return "Specific humidity (kg/kg)"
        elif var == "a2x3h_Sa_pbot":
            return "Surface pressure (Pa)"
        elif var == "a2x3h_Faxa_rainc":
            return "Convective precipitation rate (m/s)"
        elif var == "a2x3h_Faxa_rainl":
            return "Large-scale precipitation rate (m/s)"
        elif var == "a2x3h_Faxa_swndr":
            return "Downward solar radiation flux (W/m²)"
        elif var == "a2x3h_Faxa_swvdr":
            return "Downward visible solar radiation flux (W/m²)"
        elif var == "a2x3h_Faxa_swndf":
            return "Downward near-infrared solar radiation flux (W/m²)"
        elif var == "a2x3h_Faxa_swvdf":
            return "Downward visible near-infrared solar radiation flux (W/m²)"
        elif var == "windspeed":
            return "Wind speed (m/s)"
        else:
            return var

    @staticmethod
    def plot_map(var, time, zmin=None, zmax=None):
        """
        Plot data for a given variable at a given time on a map.

        Args:
            var (str): variable name
            time (str): time specification, which can be a specific time, a specific day, or a specific month
            zmax (float): maximum value for the color scale
            zmin (float): minimum value for the color scale
        """
        data = waccmdaily.get(var, time)
        if data is not None:
            fig, ax = plt.subplots(
                figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.coastlines()

            cmap = "plasma" if var == "windspeed" else "viridis"

            cf = ax.pcolormesh(
                data.lon,
                data.lat,
                data.values,
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=zmin,
                vmax=zmax,
            )

            if len(time) == 5:
                month = int(time[3:5])
                year = int(time[:2])
                plt.title(
                    f"{waccmdaily.label(var)} in {waccm.month_name(month)} of Year {year} ({waccm.months_since_nw(year, month)} months since nuclear war)"
                )
            else:
                plt.title(f"{waccmdaily.label(var)} at {time}")

            # Adjust colorbar to match the plot height
            cbar = plt.colorbar(
                cf,
                ax=ax,
                shrink=0.7,
                orientation="vertical",
                label=waccmdaily.label(var),
            )

            plt.tight_layout()
            plt.show()
