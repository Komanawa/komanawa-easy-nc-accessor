import unittest
import numpy as np
from pathlib import Path
import netCDF4 as nc
import tempfile
import matplotlib.pyplot as plt
from komanawa.easy_nc_accessor import TileIndexAccessor, CompressedSpatialAccessor

example_data_dir = Path(__file__).parents[1].joinpath('examples/dummy_dataset')
example_inputs = Path(__file__).parents[1].joinpath('examples/example_inputs')


class TestTileIndexAccessor(unittest.TestCase):

    def test_make_tile_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            accessor = TileIndexAccessor(data_dir=example_data_dir, save_index_path=tmpdir.joinpath('index.hdf'))
            data = accessor.get_index()

            assert (data['tile_number'] == [559, 560, 582, 583]).all()
            expect_bounds = np.array([[1306918., 4998423., 1356918., 5048423.],
                                      [1356918., 4998423., 1406918., 5048423.],
                                      [1306918., 4948423., 1356918., 4998423.],
                                      [1356918., 4948423., 1406918., 4998423.]])
            assert np.allclose(data[['tile_xmin', 'tile_ymin', 'tile_xmax', 'tile_ymax']].values, expect_bounds)
            assert all(data['start_date'] == '2000-07-01')
            assert all(data['end_date'] == '2003-06-30')
            expect_paths = ['559_2000-07-01_2003-06-30.nc', '560_2000-07-01_2003-06-30.nc', '582_2000-07-01_2003-06-30.nc', '583_2000-07-01_2003-06-30.nc']
            got_paths = [e.name for e in data['tile_path']]
            assert got_paths == expect_paths

    def test_get_tiles_from_extent(self):
        raise NotImplementedError()  # todo

    def test_get_tiles_from_shapefile(self):
        raise NotImplementedError()  # todo


class TestCompressedSpatialAccessor(unittest.TestCase):


    def test_plot_2d(self):
        # keynote - this is just a test that the plotting works
        nc_path = example_data_dir.joinpath('559_2000-07-01_2003-06-30.nc')
        accessor = CompressedSpatialAccessor(nc_path)

        with nc.Dataset(nc_path) as ds:
            data = np.array(ds.variables['rainfall'][:, :,1])
        data = np.nanmean(data, axis=0) * 365
        fig1, ax = accessor.plot_2d(accessor.spatial_1d_to_spatial_2d(data), base_map_path=example_inputs.joinpath('nz-topo250-maps.jpg'),
                                   title='rainfall (mm/yr)')
        fig2, ax = plt.subplots()
        fig, ax = accessor.plot_2d(accessor.spatial_1d_to_spatial_2d(data), base_map_path=example_inputs.joinpath('nz-topo250-maps.jpg'),
                                   title='rainfall (mm/yr)', vmin='10th', vmax='90th', ax=ax, contour=True,
                                   label_contours=True, contour_levels=20)
        assert fig is fig2
        if show_plots:
            accessor.show()
        else:
            accessor.close(fig1)
            accessor.close(fig2)

    # todo plot the data to check it makes sense... then make connor do the rest.




show_plots = False
if __name__ == '__main__':
    unittest.main()