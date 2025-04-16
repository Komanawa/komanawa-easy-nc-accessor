"""
created matt_dumont 
on: 4/13/25
"""
from komanawa.easy_nc_accessor import TileIndexAccessor
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

"""
This script demonstrates how to use the TileIndexAccessor class to get tiles from a shapefile or an extent.
"""

def get_tiles_from_extent():
    # lists of nztm x and nztm y coordinates define limits of the extent
    xs = [1305169, 1368943]
    ys = [4977328, 5045163]
    # define path to the directory containing the data
    data_path = Path(__file__).parent.joinpath("dummy_dataset")
    # define path of where to save the index
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_extent.hdf'
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_path, save_index_path=save_path)
    # get the tiles from the extent
    tiles = accessor.get_tiles_from_extent(xs, ys)
    accessor.get_index()
    # plot the tiles
    _plot_tiles(tiles, 'tiles from extent', extent=(xs[0], xs[1], ys[0], ys[1]))

def get_tiles_from_shapefile():
    # define paths to shapefile
    shapefile_path = Path(__file__).parent.joinpath('example_inputs/example_polygon.shp')
    # define path to the directory containing the data
    data_dir = Path(__file__).parent.joinpath('dummy_dataset')
    # define path of where to save the index (make sure you make the directory for this to sit in)
    save_path = '/home/connor/unbacked/easy-nc-accessor/index_for_shapefile.hdf'
    # create the TileIndexAccessor object
    accessor = TileIndexAccessor(data_dir=data_dir, save_index_path=save_path)
    # get the tiles from the shapefile
    tiles = accessor.get_tiles_from_shapefile(shapefile_path)
    accessor.get_index()
    # plot the tiles
    _plot_tiles(tiles, 'tiles from shapefile', shapefile_path=shapefile_path)

def _plot_tiles(tiles, name, extent=None, shapefile_path=None):
    f, ax = plt.subplots(figsize=(8, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, tile in tiles.iterrows():
        # plot the tile
        rect = plt.Rectangle((tile['tile_xmin'], tile['tile_ymin']),
                             tile['tile_xmax']-tile['tile_xmin'],
                             tile['tile_ymax']-tile['tile_ymin'],
                             alpha=0.3, color=colors[idx])
        ax.add_patch(rect)
        # plot the tile name
        ax.text((tile['tile_xmax']+tile['tile_xmin'])/2, (tile['tile_ymax']+tile['tile_ymin'])/2, f'Tile {tile['tile_number']}', fontsize=8, va='center', ha='center')

    assert not (extent is not None and shapefile_path is not None), "Only one of extent or shapefile_path should be provided"
    if shapefile_path is not None:
        gdf = gpd.read_file(shapefile_path)
        # plot the shapefile
        gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
        # plot the shapefile name
        ax.text(gdf.geometry.centroid.x.values[0], gdf.geometry.centroid.y.values[0], 'Shapefile', fontsize=8, va='center', ha='center')
    elif extent is not None:
        # plot the extent
        rect = plt.Rectangle((extent[0], extent[2]),
                             extent[1]-extent[0],
                             extent[3]-extent[2],
                             alpha=1, facecolor='none', edgecolor='black', linewidth=0.5, zorder=10)
        ax.add_patch(rect)
        # plot the extent name
        ax.text((extent[1]+extent[0])/2, (extent[3]+extent[2])/2, 'Extent', fontsize=8, va='center', ha='center')

    plt.ylabel('NZTM Y')
    plt.xlabel('NZTM X')
    # set the limits of the plot to the extent of the tiles
    plt.xlim(tiles['tile_xmin'].min(), tiles['tile_xmax'].max())
    plt.ylim(tiles['tile_ymin'].min(), tiles['tile_ymax'].max())
    ax.set_aspect('equal')
    plt.title(name)
    plt.show()

if __name__ == '__main__':
    get_tiles_from_extent()
    get_tiles_from_shapefile()