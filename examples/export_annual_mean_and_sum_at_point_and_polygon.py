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

    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'export_annual_mean_and_sum_at_point_and_polygon')
    outdir.mkdir(parents=True, exist_ok=True)

    # import point
    from example_inputs import example_point
    # set the file path. This could also be found using the TileIndexAccessor class, see example get_relevant_tiles.py
    tile_path = Path(__file__).parent.joinpath('dummy_dataset/559_2000-07-01_2003-06-30.nc')
    # create data accessor
    data_accessor = CompressedSpatialAccessor(tile_path)

    # find the location of the point in the dataset
    spatial_index = data_accessor.get_closest_loc_to_point(example_point[0], example_point[1])

    # load the dataset from the netCDF file
    with nc.Dataset(tile_path) as ds:  # this is a context manager that will close the dataset when done

        # print the number of days amalgamated per timestep... here data is presented as 7 day averages
        print(ds.ndays_amalg, 'day averages')

        # extract the rainfall data at the point. Note that the third dimension is selecting the mean (1)
        time_series_at_point = ds['rainfall'][:, spatial_index, 1]

        # get the dates from the dataset
        time_var = ds['time']
        units = time_var.units  # days since 1970-01-01 00:00:00
        # the pandas way to get the dates, this is much easier to work with, but requires human observation of the date units
        dates = (pd.to_datetime('1970-01-01') + pd.to_timedelta(time_var[:], unit='D')).round('D')

    # create a dataframe with the dates and rainfall data
    df = pd.DataFrame({'rainfall': time_series_at_point}, index=dates)

    # resample the data to annual, as find the sum and mean
    temp = df.resample('12MS').mean()
    temp = pd.DataFrame(temp)
    outdata = pd.DataFrame({
        'mean_rainfall': temp['rainfall'],
        'total_rainfall': temp['rainfall'] * 365.25
        # recall that the data is saved every 7 days so it is easiest to convert to annual from the mean.
    }, index=temp.index)

    save_path = outdir.joinpath('annual_rainfall_at_point.csv')
    # write to csv file
    outdata.to_csv(save_path, index_label='water_year_start')
    print('Exported annual rainfall to', save_path)


def export_annual_sum_and_mean_in_polygon():
    """
    This is an example of how to calculate the annual sum and mean rainfall in a polygon
    using the CompressedSpatialAccessor class.
    """
    # define the shapefile path
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/middlemarch.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')
    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'export_annual_mean_and_sum_at_point_and_polygon')
    outdir.mkdir(parents=True, exist_ok=True)

    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = outdir.joinpath('index_for_shapefile.hdf')
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile. In this case for Middlemarch the shapefile is only in one tile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)

    # iterate over the tiles (only one tile in this case)
    dates = None  # used later as part of the checking dates are the same for all tiles
    outdata = []
    for idx, tile in tiles.iterrows():
        # create data accessor object, and load the dataset from the netCDF file
        data_accessor = CompressedSpatialAccessor(tile['tile_path'])
        gridspace = data_accessor.grid_space  # get the grid space of the tile (size of each grid cell in m)

        # find the area in the polygon. Note that == 1 refers to the id of the polygon in the shapefile
        in_polygon = np.isfinite(data_accessor.shapefile_to_spatial_2d(shapefile_path, 'id', alltouched=True))
        # convert the in_polygon boolean array to 1d
        in_polygon_1d = data_accessor.spatial_2d_to_spatial_1d(in_polygon)

        with nc.Dataset(tile['tile_path']) as ds:  # use context manager to ensure file is closed after use
            # print the number of days amalgamated per timestep... here data is presented as 7 day averages
            print(ds.ndays_amalg, 'day averages')

            # get the dates from the dataset
            time_var = ds['time']
            units = time_var.units  # days since 1970-01-01 00:00:00
            # the pandas way to get the dates, this is much easier to work with, but requires human observation of the date units
            dates = (pd.to_datetime('1970-01-01') + pd.to_timedelta(time_var[:], unit='D')).round('D')

            # get the mean rainfall prediction (third dimension is selecting the mean (1))
            rainfall_data_1d = ds['rainfall'][:, :, 1]
            # set masked values to nan
            rainfall_data_1d[rainfall_data_1d.mask] = np.nan
            rainfall_data_1d = np.array(
                rainfall_data_1d)  # convert from a masked numpy array to a regular numpy array -- Note this is after flagging the masked values as  np.nan
            keep_rainfall_data = rainfall_data_1d[:, in_polygon_1d]
            if keep_rainfall_data.ndim == 1:
                keep_rainfall_data = keep_rainfall_data[:,
                                     np.newaxis]  # to allow for concatenation array dimension=(time, location(s))
            outdata.append(keep_rainfall_data)

        outdata = np.concatenate(outdata, axis=1)  # concatenate the data from all tiles on the spatial dimension

        mean_rainfall = np.nanmean(outdata, axis=1)  # across the spatial dimension
        total_rainfall = np.sum(outdata, axis=1)  # across the spatial dimension
        total_rainfall = total_rainfall / 1000 * gridspace ** 2  # convert from mm/day to m3/day

        # convert to df
        outdata_summary = pd.DataFrame({'mean_rainfall': mean_rainfall, 'total_rainfall': total_rainfall},
                                       index=dates)
        # resample the data to each water year
        outdata_summary = outdata_summary.resample('12MS').agg({'mean_rainfall': 'mean', 'total_rainfall': 'mean'})

        outdata_summary = outdata_summary * 365.25  # convert from /day to /year

        outdata_summary = outdata_summary.rename(columns={'mean_rainfall': 'Mean annual rainfall (mm/year)',
                                                          'total_rainfall': 'Total annual rainfall (m3/year)'})
        outdata_summary.to_csv(outdir.joinpath('annual_rainfall_in_polygon.csv'), index_label='water_year_start')


if __name__ == "__main__":
    export_annual_sum_and_mean_at_point()
    export_annual_sum_and_mean_in_polygon()
