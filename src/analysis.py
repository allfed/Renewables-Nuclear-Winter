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


def map_plot(var, year, month, lev=None):
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


def map_plot_diff(var, year, month, year_base=16, relative=False):
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
    """
    month_base = month
    data_current = get(var, year, month).isel(time=0).load()
    data_base = get(var, year_base, month_base).isel(time=0).load()
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


def label(var, relative=False):
    """
    Get a label for a given variable.

    Args:
        var (str): variable name
        relative (bool): whether to plot relative difference

    Returns:
        str: label
    """
    if var == "FSDS" and relative:
        return "Change in downwelling solar flux at surface (%)"
    elif var == "FSDS":
        return "Downwelling solar flux at surface (W/m²)"
    if var == "CRSOOTMR":
        return "Soot mass mixing ratio (kg/kg)"
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
