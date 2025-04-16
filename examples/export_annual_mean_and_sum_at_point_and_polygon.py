"""
created matt_dumont 
on: 4/13/25
"""
import netCDF4 as nc
from pathlib import Path

import numpy as np

from komanawa.easy_nc_accessor import CompressedSpatialAccessor
import pandas as pd
from komanawa.easy_nc_accessor import TileIndexAccessor

# todo annual / water_year mean of data at a point --> export to csv, export sum in a polygon, and make rasters for each year
def export_annual_sum_and_mean_at_point():
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
    time_series_at_point = ds['rainfall'][:, spatial_index, 1]
    # create array of dates
    dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                          freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1]
    df = pd.DataFrame({'rainfall': time_series_at_point}, index=dates)
    # amalgamate the data to annual
    temp = df.resample('12MS').agg(['sum', 'mean'])
    save_path = '/home/connor/unbacked/easy-nc-accessor/annual_rainfall_at_point.csv'
    temp.to_csv(save_path, index_label='water_year_start')

def export_annual_sum_and_mean_in_polygon():
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/middlemarch.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')
    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_shapefile.hdf'
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)

    for idx, tile in tiles.iterrows():
        data_accessor  = CompressedSpatialAccessor(tile['tile_path'])
        ds = nc.Dataset(tile['tile_path'])

        dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                              freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1]

        # get the rainfall data from each dataset
        rainfall_data_1d = ds['rainfall'][:, :, 1]
        # convert to df
        temp_df = pd.DataFrame(rainfall_data_1d, index=dates)
        # resample
        sum = temp_df.resample('12MS').sum()
        mean = temp_df.resample('12MS').mean()
        in_polygon = data_accessor.shapefile_to_spatial_2d(shapefile_path, 'id') == 1
        in_polygon_1d = data_accessor.spatial_2d_to_spatial_1d(in_polygon)
        sum.iloc[:, ~in_polygon_1d] = np.nan
        mean.iloc[:, ~in_polygon_1d] = np.nan
        sum_2d = data_accessor.spatial_1d_to_spatial_2d(sum)
        mean_2d = data_accessor.spatial_1d_to_spatial_2d(mean)
        # save data to raster
        for idx, year in enumerate(sum.index):
            save_dir = Path("/home/connor/unbacked/easy-nc-accessor/")
            data_accessor.spatial_2d_to_raster(save_dir.joinpath(f"{tile["tile_name"]}_sum_water_year_start_{year}"}), sum2d.iloc[idx])
            mean_2d.to_raster(year.strftime('%Y'), tile['tile_path'].parent, 'mean')
        pass


if __name__ == "__main__":
    # export_annual_sum_and_mean_at_point()
    export_annual_sum_and_mean_in_polygon()