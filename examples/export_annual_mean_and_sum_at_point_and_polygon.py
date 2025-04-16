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

def export_annual_sum_and_mean_at_point():
    """
    This is an example of how to export the annual sum and mean rainfall at a point
    from a netCDF file using the CompressedSpatialAccessor class.
    """

    # define path to the directory containing the data
    data_path = Path(__file__).parent.joinpath("dummy_dataset")
    # define path of where to save the index
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_extent.hdf'
    # import point
    from example_inputs import example_point
    # set the file path. This could also be found using the TileIndexAccessor class, see example get_relevant_tiles.py
    tile_path = Path(__file__).parent.joinpath('dummy_dataset/559_2000-07-01_2003-06-30.nc')
    # create data accessor
    data_accessor = CompressedSpatialAccessor(tile_path)
    # load the dataset from the netCDF file
    ds = nc.Dataset(tile_path)

    # find the location of the point in the dataset
    spatial_index = data_accessor.get_closest_loc_to_point(example_point[0], example_point[1])
    # extract the rainfall data at the point. Note that the third dimension is selecting the mean (1)
    time_series_at_point = ds['rainfall'][:, spatial_index, 1]
    # create array of dates
    dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                          freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1]
    # create a dataframe with the dates and rainfall data
    df = pd.DataFrame({'rainfall': time_series_at_point}, index=dates)
    # resample the data to annual, as find the sum and mean
    temp = df.resample('12MS').agg(['sum', 'mean'])
    save_path = '/home/connor/unbacked/easy-nc-accessor/annual_rainfall_at_point.csv'
    # write to csv file
    temp.to_csv(save_path, index_label='water_year_start')

def export_annual_sum_and_mean_in_polygon():
    """
    This is an example of how to calculate the annual sum and mean rainfall in a polygon
    and export the results to a raster file using the CompressedSpatialAccessor class.
    """
    # define the shapefile path
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/middlemarch.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')
    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_shapefile.hdf'
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile. In this case for Middlemarch the shapefile is only in one tile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)

    # iterate over the tiles (only one tile in this case)
    for idx, tile in tiles.iterrows():
        # create data accessor object, and load the dataset from the netCDF file
        data_accessor  = CompressedSpatialAccessor(tile['tile_path'])
        ds = nc.Dataset(tile['tile_path'])

        # create array of dates
        dates = pd.date_range(start=getattr(ds, 'start_date', None), end=getattr(ds, 'end_date', None),
                              freq=f'{getattr(ds, 'ndays_amalg', None)}D', inclusive='left')[:-1]

        # get the mean rainfall prediction
        rainfall_data_1d = ds['rainfall'][:, :, 1]
        # convert to df
        temp_df = pd.DataFrame(rainfall_data_1d, index=dates)
        # resample the data to each water year, as find the sum and mean
        sum = temp_df.resample('12MS').sum()
        mean = temp_df.resample('12MS').mean()
        # find the area in the polygon. Note that == 1 refers to the id of the polygon in the shapefile
        in_polygon = data_accessor.shapefile_to_spatial_2d(shapefile_path, 'id') == 1
        # convert the in_polygon boolean array to 1d
        in_polygon_1d = data_accessor.spatial_2d_to_spatial_1d(in_polygon)
        # set the values outside the polygon to NaN
        sum.iloc[:, ~in_polygon_1d] = np.nan
        mean.iloc[:, ~in_polygon_1d] = np.nan
        save_dir = Path("/home/connor/unbacked/easy-nc-accessor/")

        # save each water year start as a raster
        for idx, date in enumerate(sum.index):
            # convert the 1d array to a 2d array
            sum_2d = data_accessor.spatial_1d_to_spatial_2d(sum.iloc[idx, :])
            mean_2d = data_accessor.spatial_1d_to_spatial_2d(mean.iloc[idx, :])
            # save the 2d array as a raster
            data_accessor.spatial_2d_to_raster(save_dir.joinpath(f"{tile["tile_number"]}_sum_water_year_start_{date}"), sum_2d)
            data_accessor.spatial_2d_to_raster(save_dir.joinpath(f"{tile["tile_number"]}_mean_water_year_start_{date}"), mean_2d)


if __name__ == "__main__":
    export_annual_sum_and_mean_at_point()
    export_annual_sum_and_mean_in_polygon()