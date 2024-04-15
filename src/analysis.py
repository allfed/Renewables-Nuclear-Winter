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
import geopy
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


class GEM:
    def __init__(self):
        self.solar_farms = self.load_farms(energy="solar")
        self.wind_farms = self.load_farms(energy="wind")
        self.solar_maxyear = 15
        self.wind_maxyear = 12

    def load_farms(self, energy):
        """
        Load solar/wind farm data from Global Energy Monitor.

        Only projects that are operating, announced, in construction, or in pre-construction are included.

        Args:
            energy (str): "solar" or "wind"

        Returns:
            pd.DataFrame: solar farm data
        """
        file_path = f"../data/global-energy-monitor/Global-{energy.capitalize()}-Power-Tracker-December-2023.xlsx"
        largescale = pd.read_excel(file_path, sheet_name=1)
        mediumscale = pd.read_excel(file_path, sheet_name=2)
        if energy == "solar":
            smallscale = pd.read_excel(file_path, sheet_name=3)
            combined_df = pd.concat(
                [largescale, mediumscale, smallscale], ignore_index=True
            )
        else:
            combined_df = pd.concat([largescale, mediumscale], ignore_index=True)
        combined_df = combined_df[combined_df["Status"] != "cancelled"]
        combined_df = combined_df[combined_df["Status"] != "shelved"]
        combined_df = combined_df[combined_df["Status"] != "retired"]
        combined_df = combined_df[combined_df["Status"] != "mothballed"]
        return combined_df[
            ["Country", "Latitude", "Longitude", "Capacity (MW)", "Status"]
        ]

    def sum_operating_farms_per_country(self, energy):
        """
        Returns the total power operating capacity for the top 5 countries with solar farms.

        Useful for validation.

        Args:
            energy (str): "solar" or "wind"

        Returns:
            pd.Series: total power operating capacity for the top 5 countries with solar farms
        """
        df = self.solar_farms if energy == "solar" else self.wind_farms
        return (
            df[df.Status == "operating"]
            .groupby("Country")["Capacity (MW)"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

    def plot_farm_map(self, energy):
        """
        Plot the location of solar farms on a map.

        Args:
            energy (str): "solar" or "wind"
        """
        df = self.solar_farms if energy == "solar" else self.wind_farms
        color = "orange" if energy == "solar" else "blue"
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
            df.Longitude,
            df.Latitude,
            s=0.01 * df["Capacity (MW)"],
            color=color,
            alpha=0.5,
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
            np.array: baseline values for each month
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
    def get_wind_power_time_series(lat, lon):
        """
        Get the time series of wind power at a given location. The power is normalized by the
        baseline power at the same location, so that 1 is the baseline and 0 is no power.

        Args:
            lat (float): latitude
            lon (float): longitude

        Returns:
            np.array: time
            np.array: series of power (1 is the baseline, 0 is no power)
            np.array: seasonal variation of power (baseline)
        """
        time = []
        series = []
        baseline_values = {}
        lon = (lon + 360) % 360

        # calculate baseline using control simulation an averaging over the years
        # to get a more robust estimate of the seasonal cycle
        for month in range(1, 13):
            monthly_power = []
            for year in range(1, 11):
                data = waccmwind.get(year, month, sim="control", var="windpower_simple")
                power_output = data.sel(
                    a2x3h_ny=lat, a2x3h_nx=lon, method="nearest"
                ).windpower.values
                monthly_power.append(power_output)
            baseline_values[month] = sum(monthly_power) / len(monthly_power)

        for year in range(1, 14):
            for month in range(1, 13):
                time.append(waccm.months_since_nw(year, month))
                data = waccmwind.get(
                    year, month, sim="catastrophe", var="windpower_simple"
                )
                data_value = data.sel(
                    a2x3h_ny=lat, a2x3h_nx=lon, method="nearest"
                ).windpower.values
                series.append(data_value / baseline_values[month])
        time = np.array(time)
        series = np.array(series)
        baseline = np.array(list(baseline_values.values()))
        baseline = baseline / baseline.sum()
        return time, series, baseline

    def get_power_for_farm(self, row, energy):
        """
        Helper function to compute the sun/wind power time series for a single wind farm.
        This function will be called in parallel for each wind farm.
        """
        lat = row["Latitude"]
        lon = row["Longitude"]
        capacity = row["Capacity (MW)"]
        if energy == "solar":
            time, series, baseline = self.get_solar_flux_time_series(lat, lon)
        elif energy == "wind":
            time, series, baseline = self.get_wind_power_time_series(lat, lon)
        else:
            raise ValueError("energy must be 'solar' or 'wind'")
        weighted_series = series * capacity
        weighted_baseline = baseline * capacity
        return weighted_series, weighted_baseline

    @staticmethod
    def generate_random_locations(country_name, num_locations):
        """
        Generates a pandas DataFrame containing random locations within a country's land area.

        Args:
            country_name (str): The name of the country.
            num_locations (int): The number of random locations to generate.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'latitude' and 'longitude'.
        """

        # Fetch land polygons (for reference)
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        land = world[world.continent != "Antarctica"]

        # Initialize geolocator
        geolocator = Nominatim(user_agent="my_application")

        # Get country boundaries
        location = geolocator.geocode(country_name)
        if location is None:
            raise ValueError(f"Invalid country name: {country_name}")

        # Fetch precise country polygon
        countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        if country_name == "Democratic Republic of the Congo":
            country_name = "Dem. Rep. Congo"
        if country_name == "Cote d'Ivoire":
            country_name = "Côte d'Ivoire"
        if country_name == "Brunei Darussalam":
            country_name = "Brunei"
        if country_name == "Lybia":
            country_name = "Libya"
        if country_name == "Tanazania, United Republic":
            country_name = "Tanzania"
        if country_name == "Central African Republic":
            country_name = "Central African Rep."
        if country_name == "The Gambia":
            country_name = "Gambia"
        if country_name == "Swaziland":
            country_name = "eSwatini"
        if country_name == "South Sudan":
            country_name = "S. Sudan"
        country_polygon = countries[countries.name == country_name].geometry.iloc[0]

        # Generate random points, filtering for land
        data = []
        while len(data) < num_locations:
            x = generate_random_coordinate(
                country_polygon.bounds[0], country_polygon.bounds[2]
            )
            y = generate_random_coordinate(
                country_polygon.bounds[1], country_polygon.bounds[3]
            )
            point = Point(x, y)

            if (
                country_polygon.contains(point)
                and land.geometry.contains(point).any()
                and y < 65
            ):
                data.append({"Latitude": y, "Longitude": x, "Capacity (MW)": 1})

        return pd.DataFrame(data)

    def get_country_power_time_series(self, country, energy):
        """
        Average solar/wind power variation compared to baseline for a given country.
        We are averaging over the locations of all solar/wind farms in the country,
        weighted by their capacity.

        Args:
            country (str): country name
            energy (str): "solar" or "wind"

        Returns:
            np.array: time
            np.array: country-aggregated series of solar power variation compared to baseline
                (1 is the baseline, 0 is no solar power)
        """
        if energy == "solar":
            try:  # get location of actual solar farms
                df = self.solar_farms[self.solar_farms.Country == country]
                country_aggregated_series = np.zeros(180)
                if df.empty:
                    raise ValueError(f"No solar farms found in {country}")
            except:  # if that fails, get random location in the country
                df = self.generate_random_locations(country, 100)
                country_aggregated_series = np.zeros(180)
        elif energy == "wind":
            try:
                df = self.wind_farms[self.wind_farms.Country == country]
                country_aggregated_series = np.zeros(156)
                if df.empty:
                    raise ValueError(f"No wind farms found in {country}")
            except:
                df = self.generate_random_locations(country, 100)
                country_aggregated_series = np.zeros(156)
        else:
            raise ValueError("energy must be 'solar' or 'wind'")

        total_capacity = df["Capacity (MW)"].sum()

        country_aggregated_baseline = np.zeros(12)
        pbar = tqdm.tqdm(total=len(df))

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_power_for_farm, row, energy): row[
                    "Capacity (MW)"
                ]
                for _, row in df.iterrows()
            }

            for future in as_completed(futures):
                weighted_series, weighted_baseline = future.result()
                country_aggregated_series += weighted_series / total_capacity
                country_aggregated_baseline += weighted_baseline / total_capacity
                pbar.update(1)
        pbar.close()

        if energy == "solar":
            time, _, _ = self.get_solar_flux_time_series(0, 0)
        else:
            time, _, _ = self.get_wind_power_time_series(0, 0)

        return time, country_aggregated_series, country_aggregated_baseline

    def get_all_country_time_series(
        self,
        energy,
    ):
        """
        Calculates the average solar power variation compared to baseline for all countries.

        Output is saved to a CSV file.

        Args:
            energy (str): "solar" or "wind"
        """
        output_csv1 = f"../results/fraction_of_{energy}_power_countries.csv"
        output_csv2 = f"../results/baseline_seasonality_{energy}_power_countries.csv"
        gem_df = self.solar_farms if energy == "solar" else self.wind_farms

        # Sort countries alphabetically
        countries = sorted(gem_df["Country"].unique())

        # Add a few countries that are not in the dataset
        if energy == "solar":
            missing_countries = [
                "Brunei Darussalam",
                # "Cote d'Ivoire",
                "Iceland",
                "Luxembourg",
                # "Malta",
                "Moldova",
                "Norway",
                "Slovenia",
                "Sudan",
                "Switzerland",
                "Tajikistan",
                "Tanzania",
                "Turkmenistan",
                "Venezuela",
                "Papua New Guinea",
                "The Gambia",
                "Swaziland",
            ]
            countries.extend(missing_countries)
        elif energy == "wind":
            missing_countries = [
                "Armenia",
                # "Bahrain",
                "Benin",
                "Bhutan",
                "Botswana",
                "Brunei Darussalam",
                "Cambodia",
                "Congo",
                "Democratic Republic of the Congo",
                "Cote d'Ivoire",
                "Eritrea",
                "Gabon",
                "Haiti",
                "Iceland",
                "North Korea",
                "Lybia",
                "Malaysia",
                # "Malta",
                "Moldova",
                "Nepal",
                "Paraguay",
                "Qatar",
                "Syria",
                "Tajikistan",
                "Tanazania, United Republic",
                "Togo",
                "Trinidad and Tobago",
                "Turkmenistan",
                "Venezuela",
                "Papua New Guinea",
                "Somalia",
                "Pakistan",
                "Lybia",
                "Central African Republic",
                "Suriname",
                "The Gambia",
                "Guinea-Bissau",
                "Guinea",
                "Liberia",
                "Sierra Leone",
                "Burkina Faso",
                "Burundi",
                "Lesotho",
                "Swaziland",
                "Afghanistan",
                "South Sudan",
                "Rwanda",
                "Ghana",
            ]
            countries.extend(missing_countries)

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
            time, series, baseline = self.get_country_power_time_series(country, energy)

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

    def postprocess_aggregate_yearly(self, fraction_csv, baseline_csv, energy):
        """
        Aggregates data yearly to get the total power reduction time series for each country
        using a yearly average, where seasonality is taken into consideration.

        Args:
            fraction_csv (str): path to the CSV file containing the fraction of power over time,
                created by get_all_country_solar_power_time_series
            baseline_csv (str): path to the CSV file containing the baseline seasonality,
                created by get_all_country_solar_power_time_series
            energy (str): "solar" or "wind"

        Returns:
            None, but displays a plot
        """
        fraction_df = pd.read_csv(fraction_csv)
        baseline_df = pd.read_csv(baseline_csv)
        countries = list(fraction_df.columns[1:])
        new_df = pd.DataFrame(columns=["Year", "Country", "Fraction"])
        for country in countries:
            year_total = 0
            imonth = 0
            iyear = 0
            for month in range(len(fraction_df[country])):
                year_total += (
                    fraction_df[country][imonth] * baseline_df[country][imonth % 12]
                )
                imonth += 1
                if imonth % 12 == 0:
                    new_row = pd.DataFrame(
                        {
                            "Year": [iyear],
                            "Country": [country],
                            "Fraction": [year_total],
                        }
                    )
                    new_df = pd.concat([new_df, new_row], ignore_index=True)
                    year_total = 0
                    iyear += 1
        new_df.to_csv(f"../results/aggregate_yearly_{energy}_power.csv", index=False)
        setattr(self, f"{energy}_yearly", new_df)
        return

    def postprocess_aggregate_countries(self, input_csv, energy):
        """
        Calculate a weighted mean of the solar/wind power variation for all countries.
        Weights are given by the total solar/wind power capacity of each country.

        Args:
            input_csv (str): path to the input CSV file, created by get_all_country_solar_power_time_series
            energy (str): "solar" or "wind"

        Returns:
            None, but displays a plot
        """
        df = pd.read_csv(input_csv)
        df = df.set_index("Months_after_NW")
        gem_df = self.solar_farms if energy == "solar" else self.wind_farms
        # iterate over columns with a loop
        for col in df.columns:
            df[col] = df[col] * gem_df[gem_df.Country == col]["Capacity (MW)"].sum()
        weighted_mean = df.iloc[:, :].sum(axis=1) / gem_df["Capacity (MW)"].sum()

        # make figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index / 12, weighted_mean)
        ax.set_xlabel("Years after nuclear war")
        ax.set_ylabel(f"{energy.capitalize()} power compared to baseline")
        plt.show()
        return

    def postprocess_country_map(
        self, fraction_csv_file, baseline_csv_file, energy, zmin=None, zmax=None
    ):
        """
        Makes a map of solar power reduction over the year in each country.

        Seasonality is taken into consideration when calculating the reduction over the year.

        Args:
            fraction_csv_file (str): path to the CSV file with the fraction of solar power
            baseline_csv_file (str): path to the CSV file with the baseline seasonality
            energy (str): "solar" or "wind"
            zmin (float): minimum value for the color scale
            zmax (float): maximum value for the color scale

        Returns:
            None, but displays a plot
        """
        maxyear = self.solar_maxyear if energy == "solar" else self.wind_maxyear
        for year in range(1, maxyear):
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
                "Democratic Republic of the Congo": "Dem. Rep. Congo",
                "DR Congo": "Dem. Rep. Congo",
                "Republic of the Congo": "Congo",
                "Dominican Republic": "Dominican Rep.",
                "Türkiye": "Turkey",
                "South Sudan": "S. Sudan",
                "Central African Republic": "Central African Rep.",
                "Czech Republic": "Czechia",
                "Bosnia and Herzegovina": "Bosnia and Herz.",
                "Lybia": "Libya",
                # "Côte d'Ivoire": "Cote d'Ivoire",
            }
            country_dict = {name_mapping.get(k, k): v for k, v in country_dict.items()}

            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world["reduction"] = world["name"].map(country_dict)
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            world.boundary.plot(ax=ax, linewidth=0.5, color="k")
            world.dropna(subset=["reduction"]).plot(
                column="reduction",
                ax=ax,
                legend=True,
                legend_kwds={
                    "label": f"{energy.capitalize()} power compared to baseline (%)",
                    "orientation": "vertical",
                    "shrink": 0.5,
                },
                cmap="viridis",
                vmin=zmin,
                vmax=zmax,
            )
            world[world["reduction"].isna()].plot(ax=ax, color="lightgrey")

            ax.set_axis_off()
            ax.grid(False)
            plt.title(f"Year {year} after nuclear war")

            plt.savefig(f"../results/{energy}_reduction_map_{year:02}.pdf", dpi=300)

        os.system(
            f"pdfunite ../results/{energy}_reduction_map_*.pdf ../results/{energy}_reduction_map.pdf"
        )
        os.system(f"rm ../results/{energy}_reduction_map_*.pdf")


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
    def plot_map(
        var, year, month, lev=None, region="world", vmin=None, vmax=None, save=False
    ):
        """
        Plot data for a given variable, year, and month on a map. Can calculate and plot seasonal averages.

        Args:
            var (str): variable name
            year (int): year
            month (int or str): month number or season name (DJF, MAM, JJA, SON)
            lev (float): level to plot (hPa)
            region (str): region to plot
            vmin (float): minimum value for the color scale
            vmax (float): maximum value for the color scale
            save (bool): whether to save the plot
        """
        plt.figure()
        months_dict = {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11],
        }

        def calculate_seasonal_average(var, year, season):
            """Calculates seasonal average for the given variable, adjusting year for December."""
            months = months_dict[season]
            for m in months:
                y = year if m != 12 else year - 1  # Adjust year for December
                try:
                    ans = ans + waccm.get(var, y, m).isel(
                        time=0
                    )  # Add data for each month
                except NameError:
                    ans = waccm.get(var, y, m).isel(time=0)
            return ans / 3

        if isinstance(month, str):  # season
            if var == "total_precip":
                # Construct 'total_precip' from 'PRECC' and 'PRECL'
                toplot = calculate_seasonal_average(
                    "PRECC", year, month
                ) + calculate_seasonal_average("PRECL", year, month)
            else:
                toplot = calculate_seasonal_average(var, year, month)
        else:  # single month
            if var == "total_precip":  # total precipitation
                toplot = waccm.get("PRECC", year, month) + waccm.get(
                    "PRECL", year, month
                )
            else:
                toplot = waccm.get(var, year, month)
        if var == "total_precip":
            toplot = 1000 * 30 * 24 * 3600 * toplot
        if var == "T":
            toplot = toplot - 273.15
        if lev is not None:
            # find closest level
            lev = toplot.lev.sel(lev=lev, method="nearest")
            print(f"Level: {lev.values} hPa")
            toplot = toplot.sel(lev=lev)
        if isinstance(month, str):
            toplot_cyclic, lon_cyclic = add_cyclic_point(toplot, coord=toplot.lon)
        else:
            toplot_cyclic, lon_cyclic = add_cyclic_point(
                toplot.isel(time=0), coord=toplot.lon
            )
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        cbar = "viridis"
        if var == "CRSOOTMR":
            cbar = "pink_r"
        if var == "total_precip" or var == "PRECC" or var == "PRECL":
            cbar = "BrBG"
        if var == "T":
            cbar = "coolwarm"
        # apply region mask
        if region == "europe":
            toplot = toplot.sel(lat=slice(30, 70), lon=slice(-20, 40))
        elif region == "north_america":
            toplot = toplot.sel(lat=slice(20, 60), lon=slice(-140, -60))
        elif region == "australia":
            toplot = toplot.sel(lat=slice(-45, -10), lon=slice(110, 160))
        elif region == "south_america":
            toplot = toplot.sel(lat=slice(-60, 20), lon=slice(-90, -30))
        toplot_cyclic, lon_cyclic = add_cyclic_point(toplot, coord=toplot.lon)
        contour = ax.contourf(
            lon_cyclic,
            toplot.lat,
            toplot_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cbar,
            levels=np.linspace(vmin, vmax, 16) if vmin is not None else 21,
        )
        plt.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            label=waccm.label(var),
        )
        month_or_season = waccm.month_name(month) if isinstance(month, int) else month
        month_middle = months_dict[month][1] if isinstance(month, str) else month
        plt.title(
            f"{month_or_season} of Year {year} ({waccm.months_since_nw(year, month_middle)} months since nuclear war)"
        )
        plt.tight_layout()
        if save:
            year = str(year).zfill(2)
            plt.savefig(f"../tmp/{var}_{year}_{month}.pdf")

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
            return "Change in temperature (%)" if relative else "Temperature (°C)"
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
        elif var == "total_precip":
            return (
                "Change in total precipitation rate (%)"
                if relative
                else "Precipitation (mm/month)"
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
            "December",
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


class waccmwind:
    """
    Used to access, process, and visualize simulation data produced by the Coupe et al. 2019
    nuclear winter simulation using the Whole Atmosphere Community Climate Model (WACCM).

    This class was made specifically to access the wind data, which is not available in the
    publicly available version of the dataset. See the Climate-Data-Importing repo.
    """

    @staticmethod
    def get(year, month, sim, var="windspeed"):
        """
        Retrieve wind data for a given year and month.

        Args:
            year (int): year, where year 1 is the year of the nuclear war
            month (int): month
            sim (str): simulation name ("control" or "catastrophe")
            var (str): variable name ("windpower" or "windspeed")

        Returns:
            xr.Dataset: wind data
        """
        year = year + 4
        filepath = f"../data/wind-data/{var}_{sim}_{year:02}.nc"
        ds = xr.open_dataset(filepath)
        return ds.sel(time=f"{year:04}-{month:02}").isel(time=0)

    @staticmethod
    def plot_map(year, month, sim, zmin=None, zmax=None, var="windpower"):
        """
        Plot wind data for a given year and month on a map.

        Args:
            year (int): year, where year 1 is the year of the nuclear war
            month (int): month
            sim (str): simulation name ("control" or "catastrophe")
            zmin (float): minimum value for the color scale
            zmax (float): maximum value for the color scale
            var (str): variable name ("windpower" or "windspeed")
        """
        data = waccmwind.get(year, month, sim, var=var)
        if data is not None:
            fig, ax = plt.subplots(
                figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.coastlines()

            if var == "windpower" or var == "windpower_simple":
                z = data.windpower
                label = "Wind power (normalized)"
            elif var == "windspeed":
                z = data.windspeed
                label = "Wind speed (m/s)"

            cf = ax.pcolormesh(
                data.a2x3h_nx,
                data.a2x3h_ny,
                z,
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap="plasma",
                vmin=zmin,
                vmax=zmax,
            )

            if sim == "control":
                plt.title(
                    f"{label} in {waccm.month_name(month)} of Year {year} (control simulation)"
                )
            else:
                if waccm.months_since_nw(year, month) < 0:
                    plt.title(
                        f"{label} in {waccm.month_name(month)} of Year {year} ({-waccm.months_since_nw(year, month)} months before nuclear war)"
                    )
                else:
                    plt.title(
                        f"{label} in {waccm.month_name(month)} of Year {year} ({waccm.months_since_nw(year, month)} months since nuclear war)"
                    )

            # Adjust colorbar to match the plot height
            cbar = plt.colorbar(
                cf,
                ax=ax,
                shrink=0.7,
                orientation="vertical",
                label=label,
            )

            plt.tight_layout()
            plt.show()


def get_bounding_polygon(bbox):
    return [
        (float(bbox[2]), float(bbox[0])),  # West, South
        (float(bbox[2]), float(bbox[1])),  # West, North
        (float(bbox[3]), float(bbox[1])),  # East, North
        (float(bbox[3]), float(bbox[0])),  # East, South
        (float(bbox[2]), float(bbox[0])),  # Close polygon
    ]


def generate_random_coordinate(min_bound, max_bound):
    return min_bound + (max_bound - min_bound) * np.random.random()
