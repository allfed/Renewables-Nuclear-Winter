from concurrent.futures import ProcessPoolExecutor, as_completed

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import dask
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
        return time, series

    def get_solar_flux_for_farm(self, row):
        """
        Helper function to compute the solar flux time series for a single solar farm.
        This function will be called in parallel for each solar farm.
        """
        lat = row["Latitude"]
        lon = row["Longitude"]
        capacity = row["Capacity (MW)"]
        time, series = self.get_solar_flux_time_series(lat, lon)
        weighted_series = series * capacity
        return weighted_series

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

        pbar = tqdm.tqdm(total=len(df))

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.get_solar_flux_for_farm, row): row["Capacity (MW)"]
                for _, row in df.iterrows()
            }

            for future in as_completed(futures):
                weighted_series = future.result()
                capacity = futures[future]
                country_aggregated_series += weighted_series / total_capacity
                pbar.update(1)
        pbar.close()

        time, _ = self.get_solar_flux_time_series(0, 0)
        return time, country_aggregated_series

    def get_all_country_solar_power_time_series(
        self, output_csv="../results/solar_power_by_country_nuclear_winter.csv"
    ):
        """
        Calculates the average solar power variation compared to baseline for all countries.

        Output is saved to a CSV file.

        Args:
            output_csv (str): path to the output CSV file
        """
        # Sort countries alphabetically
        countries = sorted(self.solar_farms["Country"].unique())

        # Check if the output file already exists to determine where to resume
        try:
            existing_df = pd.read_csv(output_csv)
            completed_countries = existing_df.columns[1:]  # Exclude 'Time' column
            countries_to_process = [
                c for c in countries if c not in completed_countries
            ]
        except FileNotFoundError:
            # If file does not exist, start from scratch
            existing_df = None
            countries_to_process = countries

        # Process each country
        for country in countries_to_process:
            print(f"Processing {country}...")
            time, series = self.get_country_solar_power_time_series(country)

            # If this is the first country being processed, initialize DataFrame
            if existing_df is None:
                existing_df = pd.DataFrame(time, columns=["Time"])
                existing_df[country] = series
            else:
                existing_df[country] = series

            # Write (or overwrite) the CSV file with updated data
            existing_df.to_csv(output_csv, index=False)
            print(f"Saved {country} to {output_csv}")


class waccm:
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