"""
created matt_dumont 
on: 4/13/25
"""
import numpy as np
from komanawa.easy_nc_accessor import CompressedSpatialAccessor
import netCDF4 as nc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries_at_point_with_uncertainty():
    """
    This is an example of how to plot the time series data at a point with uncertainty
    """

    # import point
    from example_inputs import example_point
    # tile path
    tile_path = Path(__file__).parent.joinpath('dummy_dataset/559_2000-07-01_2003-06-30.nc')
    data_accessor = CompressedSpatialAccessor(tile_path)
    # load the dataset from the netCDF file
    # find the location of the point in the dataset
    spatial_index, dist = data_accessor.get_closest_loc_to_point(example_point[0], example_point[1])

    with nc.Dataset(tile_path) as ds:  # this is a context manager that will close the dataset when done
        # get the data at the point
        time_series_data = ds['rainfall'][:, spatial_index, :]  # dimensions are (time, spatial, variance)
        time_series_data[time_series_data.mask] = np.nan  # set masked values to nan
        time_series_data = np.array(
            time_series_data)  # convert from a masked numpy array to a regular numpy array -- Note this is after flagging the masked values as  np.nan

        lower_bound = time_series_data[:, 0]
        mean = time_series_data[:, 1]
        upper_bound = time_series_data[:, 2]

        # get the dates from the dataset
        time_var = ds['time']
        units = time_var.units  # days since 1970-01-01 00:00:00

        # the nc way to get the dates, note this is not super helpful as they do not work well with pandas datetimes
        dates = nc.num2date(time_var[:], units=units, calendar='standard')

        # the pandas way to get the dates, this is much easier to work with, but requires human observation of the date units
        dates = pd.to_datetime('1970-01-01') + pd.to_timedelta(time_var[:], unit='D')

    # plot data
    f, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, mean, label='Mean', lw=0.5, zorder=10)
    ax.fill_between(dates, lower_bound, upper_bound, alpha=0.7, label='Uncertainty', color='tab:orange')
    ax.set_title('Time Series Data at Example Point')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm/day)')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot_timeseries_at_point_with_uncertainty()
