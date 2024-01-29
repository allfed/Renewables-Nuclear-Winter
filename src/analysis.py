import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


def get(var, year, month):
    """
    Get data for a given variable, year, and month.

    Year 1 is the year of the nuclear war (which starts in May).

    Args:
        var (str): variable name
        year (int): year
        month (int): month

    Returns:
        xr.DataArray: data
    """
    year = year + 4
    year = str(year).zfill(4)
    month = str(month).zfill(2)
    ds = xr.open_dataset(
        f"../data/nw_ur_150_07_mini/nw_ur_150_07.cam.h0.{year}-{month}.nc"
    )
    return ds[var]


def plot_map(var, year, month, lev=None):
    """
    Plot data for a given variable, year, and month on a map.

    Args:
        var (str): variable name
        year (int): year
        month (int): month
        lev (float): level to plot (hPa)
    """
    toplot = get(var, year, month)
    if lev is not None:
        # find closest level
        lev = toplot.lev.sel(lev=lev, method="nearest")
        print(f"Level: {lev.values} hPa")
        toplot = toplot.sel(lev=lev)
    toplot_cyclic, lon_cyclic = add_cyclic_point(toplot.isel(time=0), coord=toplot.lon)
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
        label=label(var),
    )
    plt.title(
        f"{month_name(month)} of Year {year} ({months_since_nw(year, month)} months since nuclear war)"
    )
    plt.tight_layout()


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
    data_current = get(var, year, month).isel(time=0).load()
    data_base = get(var, year_base, month_base).isel(time=0).load()
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
        label=label(var, relative=relative),
    )
    plt.title(
        f"{month_name(month)} of Year {year} ({months_since_nw(year, month)} months since nuclear war)"
    )
    plt.tight_layout()
    plt.show()


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
            time.append(months_since_nw(year, month))
            data = get(var, year, month).isel(time=0)
            ref_data = get(var, year_base, month).isel(time=0)
            data_global_average = global_average(data, lev=lev)
            ref_data_global_average = global_average(ref_data, lev=lev)
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
    plt.ylabel(label(var, relative=relative))
    plt.tight_layout()
    plt.title("Global average")


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
                time.append(months_since_nw(year, month))
                data = get(var, year, month).isel(time=0)
                ref_data = get(var, year_base, month).isel(time=0)
                data_value = data.sel(lat=lat_val, lon=lon_val, method="nearest")
                ref_data_value = ref_data.sel(lat=lat_val, lon=lon_val, method="nearest")
                if diff and relative:
                    series.append(100 * (data_value - ref_data_value) / ref_data_value)
                elif diff:
                    series.append(data_value - ref_data_value)
                else:
                    series.append(data_value)
        time = np.array(time)
        ls = ls_list.pop(0)
        plt.plot(time / 12, series, label=format_coords(lat_val, lon_val), ls=ls)
    plt.xlabel("Years since nuclear war")
    plt.ylabel(label(var, relative=relative))
    plt.tight_layout()
    plt.legend()
    plt.show()

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    for lat_val, lon_val in zip(lat, lon):
        ax.plot(lon_val, lat_val, "o", markersize=5, transform=ccrs.Geodetic())
    plt.show()


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
        return "Change in surface pressure (%)" if relative else "Surface pressure (Pa)"
    elif var == "Q":
        return (
            "Change in specific humidity (%)"
            if relative
            else "Specific humidity (kg/kg)"
        )
    elif var == "RELHUM":
        return (
            "Change in relative humidity (%)" if relative else "Relative humidity (%)"
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
