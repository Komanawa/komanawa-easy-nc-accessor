"""
created matt_dumont 
on: 4/13/25
"""
import pandas as pd
from komanawa.easy_nc_accessor import TileIndexAccessor
from komanawa.easy_nc_accessor import CompressedSpatialAccessor
import netCDF4 as nc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_and_export_timeseries_rainfall_averaged_over_polygon():
    """
    This is an example of how to calculate a timeseries of the mean rainfall in a polygon, plot the results and export as a CSV.
    """
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/example_polygon.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')

    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'plot_and_export_timeseries_rainfall_averaged_over_polygon')
    outdir.mkdir(parents=True, exist_ok=True)
    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = outdir.joinpath('index_for_shapefile.hdf')

    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)
    # create data accessor object for each tile
    average_rainfall_data_in_polygon_by_tile = []

    dates = None  # used later as part of the checking dates are the same for all tiles
    for idx, tile in tiles.iterrows():
        data_accessor = CompressedSpatialAccessor(tile['tile_path'])

        # find spatial extent in polygon
        in_polygon = data_accessor.shapefile_to_spatial_2d(shapefile_path, 'id') == 1
        in_polygon_1d = data_accessor.spatial_2d_to_spatial_1d(in_polygon)

        with nc.Dataset(tile['tile_path']) as ds:  # use context manager to ensure file is closed after use
            # get the rainfall data from each dataset
            rainfall_data_1d = ds['rainfall'][:, :, 1]
            # set masked values to nan
            rainfall_data_1d[rainfall_data_1d.mask] = np.nan  # set masked values to nan
            rainfall_data_1d = np.array(
                rainfall_data_1d)  # convert from a masked numpy array to a regular numpy array -- Note this is after flagging the masked values as  np.nan

            # get the dates from the dataset
            time_var = ds['time']
            units = time_var.units  # days since 1970-01-01 00:00:00
            # the pandas way to get the dates, this is much easier to work with, but requires human observation of the date units
            new_dates = (pd.to_datetime('1970-01-01') + pd.to_timedelta(time_var[:], unit='D')).round('D')

            # check if the dates are the same for all tiles
            if dates is None:
                dates = new_dates
            else:
                assert np.all(dates == new_dates), "Dates are not the same for all tiles"

        keep_data = rainfall_data_1d[:, in_polygon_1d]
        if keep_data.ndim == 1:
            keep_data = keep_data[:, np.newaxis]  # to allow for concatenation array dimension=(time, location(s))
        average_rainfall_data_in_polygon_by_tile.append(keep_data)

    # concatenate the data from all tiles
    average_rainfall_data_in_polygon_by_tile = np.concatenate(average_rainfall_data_in_polygon_by_tile, axis=1)

    # average the data over the spatial dimension
    average_rainfall_in_polygon = np.nanmean(average_rainfall_data_in_polygon_by_tile, axis=1)

    # plot the average rainfall in polygon
    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, average_rainfall_in_polygon, label='Mean', lw=1, zorder=10)
    ax.set_title('Average rainfall in polygon')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm/day)')
    ax.legend()
    plt.show()

    # export rainfall data to csv
    df = pd.DataFrame({'date': dates, 'rainfall': average_rainfall_in_polygon})
    csv_path = outdir.joinpath('average_rainfall_in_polygon.csv')
    df.to_csv(csv_path, index=False)
    print(f"Exported rainfall data to {csv_path}")


if __name__ == "__main__":
    plot_and_export_timeseries_rainfall_averaged_over_polygon()
