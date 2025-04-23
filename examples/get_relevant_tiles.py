"""
created matt_dumont 
on: 4/13/25
"""
from komanawa.easy_nc_accessor import TileIndexAccessor
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path


def export_tile_bounds_to_shapefile():
    """
    This is an example of how to export the tile bounds to a shapefile.
    :return:
    """
    # define path to the directory containing the data
    data_path = Path(__file__).parent.joinpath("dummy_dataset")
    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'get_relevant_tiles')
    outdir.mkdir(parents=True, exist_ok=True)
    # define path of where to save the index
    save_path = outdir.joinpath('index_for_extent.hdf')
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_path, save_index_path=save_path)
    accessor.export_tiles_to_shapefile(outdir.joinpath('tile_bounds.shp'))
    print('Exported tile bounds to', outdir.joinpath('tile_bounds.shp'))


def get_tiles_from_extent():
    """
    This is an example of how to get the tiles that intersect with a given extent.
    """

    # lists of nztm x and nztm y coordinates define limits of the extent
    xs = [1305169, 1368943]
    ys = [4977328, 5045163]
    # define path to the directory containing the data
    data_path = Path(__file__).parent.joinpath("dummy_dataset")
    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'get_relevant_tiles')
    outdir.mkdir(parents=True, exist_ok=True)
    # define path of where to save the index
    save_path = outdir.joinpath('index_for_extent.hdf')
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_path, save_index_path=save_path)
    # get the tiles from the extent
    tiles = accessor.get_tiles_from_extent(xs, ys)
    accessor.get_index()

    # plot the tiles
    basemap_path = Path(__file__).parent.joinpath('example_inputs/nz-topo250-maps.jpg')
    fig, ax = accessor.plot_tiles(tiles=tiles['tile_number'], basemap_path=basemap_path)
    ax.set_title('Tiles from extent')
    plt.show()


def get_tiles_from_shapefile():
    """
    This is an example of how to get the tiles that intersect with a given shapefile.
    """
    # define paths to shapefile
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/example_polygon.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')

    # directory to save the outputs / index -- defaults to Downloads
    outdir = Path.home().joinpath('Downloads', 'easy_nc_accessor_examples',
                                  'get_relevant_tiles')
    outdir.mkdir(parents=True, exist_ok=True)

    # define path of where to save the index
    save_path = outdir.joinpath('index_for_shapefile.hdf')
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)
    accessor.get_index()
    # plot the tiles
    basemap_path = Path(__file__).parent.joinpath('example_inputs/nz-topo250-maps.jpg')
    fig, ax = accessor.plot_tiles(tiles=tiles['tile_number'], basemap_path=basemap_path)
    ax.set_title('Tiles from shapefile')
    plt.show()


if __name__ == '__main__':
    export_tile_bounds_to_shapefile()
    get_tiles_from_extent()
    get_tiles_from_shapefile()
