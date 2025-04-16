"""
created matt_dumont 
on: 4/13/25
"""

from komanawa.easy_nc_accessor import CompressedSpatialAccessor
import netCDF4 as nc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_timeseries_at_point_with_uncertainty():
    # define path to the directory containing the data
    data_path = Path(__file__).parent.joinpath("dummy_dataset")
    # define path of where to save the index
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_extent.hdf'
    # import point
    from example_inputs import example_point
    # tile path
    tile_path = Path(__file__).parent.joinpath('dummy_dataset/559_2000-07-01_2003-06-30.nc')
    data_accessor = CompressedSpatialAccessor(tile_path)
    # load the dataset from the netCDF file
    ds = nc.Dataset(tile_path)

    # find the location of the point in the dataset
    spatial_index = data_accessor.get_closest_loc_to_point(example_point[0], example_point[1])

    # get the time series data at the point
    time_series_data = ds['rainfall'][:, spatial_index, :]
    lower_bound = time_series_data[:, 0]
    mean = time_series_data[:, 1]
    upper_bound = time_series_data[:, 2]
    # create array of dates
    dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                          freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1] # todo should this be a class function

    # plot data
    f, ax = plt.subplots(figsize=(15,6))
    ax.plot(dates, mean, label='Mean', lw=0.5, zorder=10)
    ax.fill_between(dates, lower_bound, upper_bound, alpha=0.7, label='Uncertainty', color='tab:orange')
    ax.set_title('Time Series Data at Example Point')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm/day)')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_timeseries_at_point_with_uncertainty()

