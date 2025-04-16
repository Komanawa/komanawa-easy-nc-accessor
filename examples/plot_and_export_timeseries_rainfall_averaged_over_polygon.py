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
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/example_polygon.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')
    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_shapefile.hdf'
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)
    # create data accessor object for each tile
    average_rainfall_data_in_polygon_by_tile = []
    for idx, tile in tiles.iterrows():
        data_accessor  = CompressedSpatialAccessor(tile['tile_path'])
        ds = nc.Dataset(tile['tile_path'])
        # get the rainfall data from each dataset
        rainfall_data_1d = ds['rainfall'][:, :, 1]
        # find spatial extent in polygon
        in_polygon = data_accessor.shapefile_to_spatial_2d(shapefile_path, 'id') == 1
        in_polygon_1d = data_accessor.spatial_2d_to_spatial_1d(in_polygon)
        average_rainfall_data_in_polygon_by_tile.append(rainfall_data_1d[:, in_polygon_1d].mean(axis=1))

    average_rainfall_in_polygon = np.nanmean(average_rainfall_data_in_polygon_by_tile, axis=0)
    dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                  freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1] # todo should this be a class function

    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, average_rainfall_in_polygon, label='Mean', lw=1, zorder=10)
    ax.set_title('Average rainfall in polygon')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm/day)')
    ax.legend()
    plt.show()
    # export rainfall data to csv
    df = pd.DataFrame({'date': dates, 'rainfall': average_rainfall_in_polygon})
    csv_path = '/home/connor/unbacked/easy-nc-accessor/average_rainfall_in_polygon.csv'
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    plot_and_export_timeseries_rainfall_averaged_over_polygon()
