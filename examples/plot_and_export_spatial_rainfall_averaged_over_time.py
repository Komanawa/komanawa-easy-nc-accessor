"""
created matt_dumont 
on: 4/13/25
"""
from komanawa.easy_nc_accessor import CompressedSpatialAccessor
from pathlib import Path
import netCDF4 as nc
import matplotlib.pyplot as plt

def plot_and_export_spatial_rainfall_averaged_over_time():
    """
    This is an example of how to calculate the mean rainfall over time in a tile, plot the results and export as a GeoTIFF.
    """
    # choose one of the data files to use
    tile_data_path = Path(__file__).parent.joinpath('dummy_dataset/559_2000-07-01_2003-06-30.nc')
    # create data accessor
    accessor = CompressedSpatialAccessor(tile_data_path)
    # load the dataset from the netCDF file
    ds = nc.Dataset(tile_data_path)
    # get mean rainfall over the time dimension, note that 1 refers to the average variance
    mean_rainfall = ds['rainfall'][:, :, 1].mean(axis=0)
    # transform the 1d array to a 2d array
    mean_rainfall_2d = accessor.spatial_1d_to_spatial_2d(mean_rainfall)
    # rainfall path
    base_map_path = Path(__file__).parent.joinpath('example_inputs/nz-topo250-maps.jpg')
    # plot the mean rainfall
    f, ax = accessor.plot_2d(mean_rainfall_2d, base_map_path=base_map_path, cbar_lab="Rainfall (mm/day)", alpha=0.5)
    ax.set_title('Mean Rainfall')
    ax.set_xlabel('NZTM X (m)')
    ax.set_ylabel('NZTM Y (m)')
    plt.show()
    # export the mean rainfall as a GeoTIFF
    raster_path = "/home/connor/unbacked/easy-nc-accessor/mean_rainfall_spatial_mean.tif"
    accessor.spatial_2d_to_raster(raster_path, mean_rainfall_2d)

if __name__ == "__main__":
    plot_and_export_spatial_rainfall_averaged_over_time()