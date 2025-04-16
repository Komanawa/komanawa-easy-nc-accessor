"""
created matt_dumont 
on: 4/13/25
"""
import warnings
from pathlib import Path
from osgeo import gdal, ogr, osr
import numpy as np
import netCDF4 as nc
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import from_levels_and_colors


class _common_functions:
    def check_shape_crs(self, shapefile_path=None, shapefile=None):
        """
        check the crs of a shapefile

        :param shapefile_path: path to shapefile
        :param shapefile: shapefile object (geodataframe/geoseries), note shapely geometries do not have a crs
        :return: None
        """
        if shapefile_path is not None:
            data = gpd.read_file(shapefile_path)
        elif shapefile is not None:
            data = shapefile
        else:
            raise ValueError('must define either shapefile_path or shapefile')
        assert len(data) > 0, 'no features found in shapefile'
        assert data.crs is not None, 'no crs found in shapefile'
        shp_epsg = pyproj.CRS(data.crs).to_epsg()
        if shp_epsg != self.epsg_num:
            raise ValueError(f'epsg of shapefile {shp_epsg} does not match model epsg {self.epsg_num}'
                             f'If you think this is wrong set check_crs=False, pyproj can do silly things sometimes')

    def check_raster_crs(self, raster_path):
        """
        check the crs of a raster

        :param raster_path: path to raster
        :return:
        """
        if isinstance(raster_path, Path):
            raster_path = str(raster_path)
        sourceRaster = gdal.Open(raster_path)

        in_epsg = pyproj.CRS.from_wkt(sourceRaster.GetProjectionRef()).to_epsg()
        if in_epsg != self.epsg_num:
            raise ValueError(
                f'epsg of raster {in_epsg} does not match model epsg {self.epsg_num}, got raw {sourceRaster.GetProjectionRef()}'
                f'If you think this is wrong set check_crs=False, pyproj can do silly things sometimes')
        sourceRaster = None  # kill the raster

class TileIndexAccessor(_common_functions):
    """
    Support to select geospatial tiles from a directory of netcdf files.  The netcdf files must have the following attributes:

    * xmin: the minimum x coordinate of the tile
    * ymin: the minimum y coordinate of the tile
    * xmax: the maximum x coordinate of the tile
    * ymax: the maximum y coordinate of the tile

    The netcdf files can have the following attributes:

    * tile_number: the tile number
    * start_date: the start date of the tile (inclusive)
    * end_date: the end date of the tile (inclusive)

    All NetCDF files within the directory and it's children are searched for the above attributes.

    The key mechanism is to create a tile index from the netcdf files in the directory and then return a dataframe of the tiles that fall within a bounding box, shapefile for all datasets. The end user can then do subsequent filtering based on the other parameters (e.g. tile number, start date, and end date)
    """
    epsg_num = 2193

    def __init__(self, data_dir, save_index_path):
        """
        :param data_dir: path to the directory containing the netcdf files
        :param save_index_path: path to save the index file
        """
        data_dir = Path(data_dir)
        assert data_dir.exists(), f'data_dir {data_dir} does not exist'
        assert data_dir.is_dir(), f'data_dir {data_dir} is not a directory'
        self.data_dir = data_dir

        self.save_index_path = Path(save_index_path)
        assert self.save_index_path.parent.exists(), f'save_index_path parent {self.save_index_path.parent} does not exist'
        assert self.save_index_path.suffix == '.hdf', f'save_index_path {self.save_index_path} must be a .hdf file'

    def get_index(self, recalc=False):
        """
        get / make a tile index from the netcdf files in the data_dir

        :return: dataframe of all tiles that fall within the shapefile. columns are:

            * 'tile_path': path of the tile relative to the data_dir
            * 'tile_number': the tile number.  This can be missing without causing an exception
            * 'tile_xmin': the minimum x coordinate of the tile
            * 'tile_ymin': the minimum y coordinate of the tile
            * 'tile_xmax': the maximum x coordinate of the tile
            * 'tile_ymax': the maximum y coordinate of the tile
            * 'start_date': the start date of the tile (inclusive) This can be missing without causing an exception
            * 'end_date': the end date of the tile (inclusive) This can be missing without causing an exception
        """

        if self.save_index_path.exists() and not recalc:
            data = pd.read_hdf(self.save_index_path, key='index')
            assert isinstance(data, pd.DataFrame), f'{self.save_index_path} must be a dataframe'
        else:

            # make the index
            ncfiles = list(self.data_dir.glob('**/*.nc'))
            ncfiles = list(sorted(ncfiles))
            assert len(ncfiles) > 0, f'{self.save_index_path} does not contain any .nc files, check the path'

            data = pd.DataFrame(index=range(len(ncfiles)),
                                columns=['tile_path', 'tile_number', 'tile_xmin', 'tile_ymin', 'tile_xmax', 'tile_ymax',
                                         'start_date', 'end_date'])
            for i, f in enumerate(ncfiles):
                assert f.exists(), f'{f} does not exist'
                assert f.is_file(), f'{f} is not a file'
                data.loc[i, 'tile_path'] = str(f.relative_to(self.data_dir))
                with nc.Dataset(f) as ds:
                    tnumber = getattr(ds,'tile_number', None)
                    data.loc[i, 'tile_number'] = tnumber
                    for k in ['xmin', 'ymin', 'xmax', 'ymax']:
                        try:
                            data.loc[i, f'tile_{k}'] = ds.getncattr(k)
                        except KeyError:
                            raise KeyError(f'{k} not found in {f}')
                    start_date = getattr(ds, 'start_date', None)
                    end_date = getattr(ds, 'end_date', None)
                    data.loc[i, 'start_date'] = start_date
                    data.loc[i, 'end_date'] = end_date

            data[['tile_xmin', 'tile_ymin', 'tile_xmax', 'tile_ymax']] = data[
                ['tile_xmin', 'tile_ymin', 'tile_xmax', 'tile_ymax']].astype(float)
            data[['tile_number']] = data[['tile_number']].astype(int)
            data[['start_date', 'end_date', 'tile_path']] = data[['start_date', 'end_date', 'tile_path']].astype(str)

            data.to_hdf(self.save_index_path, key='index')

        # transform the tile_path to a Path object and convert the dates to datetime objects
        data['tile_path'] = [self.data_dir.joinpath(p) for p in data['tile_path']]
        data['start_date'] = pd.to_datetime(data['start_date'])
        data['end_date'] = pd.to_datetime(data['end_date'])
        return data

    def get_tiles_from_extent(self, xs, ys):
        """
        get the tile paths from a bounding box

        :param xs: the x coordinates of the bounding box / extent.  the min/max of the passed x coordinates are used
        :param ys: the y coordinates of the bounding box / extent.  the min/max of the passed y coordinates are used
        :return: dataframe of all tiles that fall within the shapefile. columns are:

            * 'tile_path': pathlib.Path to the tile
            * 'tile_number': the tile number.  This can be missing without causing an exception
            * 'tile_xmin': the minimum x coordinate of the tile
            * 'tile_ymin': the minimum y coordinate of the tile
            * 'tile_xmax': the maximum x coordinate of the tile
            * 'tile_ymax': the maximum y coordinate of the tile
            * 'start_date': the start date of the tile (inclusive)   This can be missing without causing an exception
            * 'end_date': the end date of the tile (inclusive)   This can be missing without causing an exception
        """
        xmin = np.min(xs)
        xmax = np.max(xs)
        ymin = np.min(ys)
        ymax = np.max(ys)
        assert xmin < xmax, 'xmin must be less than xmax'
        assert ymin < ymax, 'ymin must be less than ymax'

        path_index = self.get_index()

        overlap = ((xmin <= path_index['tile_xmax'])
                   & (xmax >= path_index['tile_xmin'])
                   & (ymin <= path_index['tile_ymax'])
                   & (ymax >= path_index['tile_ymin']))

        return path_index.loc[overlap]

    def get_tiles_from_shapefile(self, shapefile_path, check_crs=True):
        """
        get the tile paths from a shapefile

        :param shapefile_path: path to the shapefile
        :param check_crs: boolean if True check the crs of the shapefile
        :return: dataframe of all tiles that fall within the shapefile. columns are:

            * 'tile_path': pathlib.Path to the tile
            * 'tile_number': the tile number.  This can be missing without causing an exception
            * 'tile_xmin': the minimum x coordinate of the tile
            * 'tile_ymin': the minimum y coordinate of the tile
            * 'tile_xmax': the maximum x coordinate of the tile
            * 'tile_ymax': the maximum y coordinate of the tile
            * 'start_date': the start date of the tile (inclusive)   This can be missing without causing an exception
            * 'end_date': the end date of the tile (inclusive)   This can be missing without causing an exception
        """

        if check_crs:
            self.check_shape_crs(shapefile_path)
        data = gpd.read_file(shapefile_path)
        minx, miny, maxx, maxy = data.total_bounds
        xs = np.array([minx, maxx])
        ys = np.array([miny, maxy])
        return self.get_tiles_from_extent(xs, ys)


class _BaseAccessor(_common_functions):
    epsg_num = 2193
    grid_space = None  # set in init
    spatial_2d_shape = None  # (rows, cols) set in init

    def __init__(self, datapath, active_index_name='active_index', grid_x_name='grid_x', grid_y_name='grid_y', loc_x_name='x', loc_y_name='y'):
        """

        :param datapath: path to the netcdf file
        :param active_index_name: name of the active index variable in the netcdf file
        :param grid_x_name: name of the grid x variable in the netcdf file
        :param grid_y_name: name of the grid y variable in the netcdf file
        :param loc_x_name: name of the x location variable in the netcdf file (to support compressed spatial dimensions)
        :param loc_y_name: name of the y location variable in the netcdf file (to support compressed spatial dimensions)
        """

        datapath = Path(datapath)
        assert datapath.exists(), f'datapath {datapath} does not exist'
        self.datapath = datapath
        try:
            with nc.Dataset(self.datapath) as ds:
                pass
        except Exception as e:
            raise ValueError(f'problem reading netcdf file {self.datapath}') from e

        with nc.Dataset(self.datapath) as ds:
            got_vars = set(ds.variables.keys())
            expected_vars = {active_index_name, grid_x_name, grid_y_name, loc_x_name, loc_y_name}
            assert expected_vars.issubset(got_vars), f'expected variables {expected_vars} not found in {got_vars}'
            self.grid_space = ds.variables[grid_x_name][1] - ds.variables[grid_x_name][0]
            self.spatial_2d_shape = ds.variables[active_index_name].shape
        self.active_index_name = active_index_name
        self.grid_x_name = grid_x_name
        self.grid_y_name = grid_y_name
        self.loc_x_name = loc_x_name
        self.loc_y_name = loc_y_name

    def get_active_index(self):
        """
        read the active index of the dataset, this is a boolean 2d array of the same shape as the spatial 2d shape.

        True = has data, False = no data

        :return:
        """
        with nc.Dataset(self.datapath) as ds:
            active_index = ds.variables[self.active_index_name][:]
            active_index = np.array(active_index, dtype=bool)
        return active_index

    def get_xlim_ylim(self):
        """
        read the dataset spatial limits

        :return:  x_min, x_max, y_min, y_max
        """
        with nc.Dataset(self.datapath) as ds:
            xs = ds.variables[self.grid_x_name][:]
            ys = ds.variables[self.grid_y_name][:]
            x_min = xs.min() - self.grid_space / 2
            x_max = xs.max() + self.grid_space / 2
            y_min = ys.min() - self.grid_space / 2
            y_max = ys.max() + self.grid_space / 2
        return x_min, x_max, y_min, y_max

    @staticmethod
    def _get_gdal_dtype(dtype):
        type_mapper = {
            float: gdal.GDT_Float64,
            int: gdal.GDT_Int32,

            'float': gdal.GDT_Float64,
            'int': gdal.GDT_Int32,

            np.float32: gdal.GDT_Float32,
            np.float64: gdal.GDT_Float64,
            np.int8: gdal.GDT_Int8,
            np.int16: gdal.GDT_Int16,
            np.int32: gdal.GDT_Int32,
            np.uint16: gdal.GDT_UInt16,
            np.uint32: gdal.GDT_UInt32,

            'float32': gdal.GDT_Float32,
            'float64': gdal.GDT_Float64,
            'int8': gdal.GDT_Int8,
            'int16': gdal.GDT_Int16,
            'int32': gdal.GDT_Int32,
            'uint16': gdal.GDT_UInt16,
            'uint32': gdal.GDT_UInt32,

            gdal.GDT_Float32: gdal.GDT_Float32,
            gdal.GDT_Float64: gdal.GDT_Float64,
            gdal.GDT_Int8: gdal.GDT_Int8,
            gdal.GDT_Int16: gdal.GDT_Int16,
            gdal.GDT_Int32: gdal.GDT_Int32,
            gdal.GDT_UInt16: gdal.GDT_UInt16,
            gdal.GDT_UInt32: gdal.GDT_UInt32,

        }
        raw_dt = dtype
        dtype = type_mapper.get(dtype, None)
        if dtype is None:
            raise ValueError(f'dtype: {raw_dt} not acceptable. Acceptable values: {type_mapper}')
        return dtype

    def get_2d_spatial_zero(self, dtype=float):
        """
        get a 2d array of zeros with the same shape as the spatial 2d shape

        :param dtype: data type of the array
        :return:
        """

        return np.zeros(self.spatial_2d_shape, dtype=dtype)

    def spatial_2d_to_raster(self, path, array, dtype=np.float32, compression=True):
        """
        saves a 2d array as a raster geotiff file

        :param path: path to save the raster
        :param array: array to save (must be 2d model array)
        :param dtype: gdal data type to save as
        :param compression: boolean if True use compression (LZW, options = 'COMPRESS=LZW', 'PREDICTOR={p}', 'TILED=YES') where p=2 for int and 3 for float
        :return:
        """
        dtype = self._get_gdal_dtype(dtype)

        if dtype in [gdal.GDT_Int8, gdal.GDT_Int16, gdal.GDT_Int32, gdal.GDT_UInt16, gdal.GDT_UInt32]:
            predictor = 2
        elif dtype in [gdal.GDT_Float32, gdal.GDT_Float64]:
            predictor = 3
        else:
            raise ValueError('dtype must be one of the acceptable_dtypes')

        if compression:
            kwargs = dict(options=['COMPRESS=LZW', f'PREDICTOR={predictor}', 'TILED=YES'])
        else:
            kwargs = dict()

        if isinstance(path, Path):
            path = str(path)  # gdal cannot handle pathlib paths
        null_val = -999999

        if array.shape != self.spatial_2d_shape:
            raise ValueError(f'{array.shape=} must match {self.spatial_2d_shape=}')
        no_flow = self.get_active_index()
        array[~no_flow] = null_val # review, hey matt, I noticed that if you export to raster and then try plot the array in python, you get this no data values
        # could be worth working on a copy of the array instead of modifying the original here?
        output_raster = gdal.GetDriverByName('GTiff').Create(path, array.shape[1], array.shape[0], 1,
                                                             dtype, **kwargs)  # Open the file
        x_min, x_max, y_min, y_max = self.get_xlim_ylim()
        temp = pyproj.CRS.from_epsg(self.epsg_num).to_wkt()
        output_raster.SetProjection(temp)
        output_raster.SetGeoTransform((x_min, self.grid_space, 0, y_min, 0, self.grid_space))
        output_raster.FlushCache()  # Exports the coordinate system
        # to the file
        band = output_raster.GetRasterBand(1)
        band.WriteArray(np.flipud(array))  # Writes my array to the raster
        band.SetNoDataValue(null_val)
        band.FlushCache()
        band = None
        output_raster = None

    def _get_grid_x_y(self, cell_centers=True):
        if cell_centers:
            with nc.Dataset(self.datapath) as ds:
                xs = np.array(ds.variables[self.grid_x_name])
                ys = np.array(ds.variables[self.grid_y_name])
        else:
            x_min, x_max, y_min, y_max = self.get_xlim_ylim()
            xs = np.linspace(x_min, x_max, self.spatial_2d_shape[1] + 1)
            ys = np.flip(np.linspace(y_min, y_max, self.spatial_2d_shape[0] + 1))
        xs, ys = np.meshgrid(xs, ys)
        return xs, ys

    def plot_2d(self, array, vmin=None, vmax=None, title=None, ax=None, color_bar=True,
                base_map_path=None,
                cbar_lab=None,
                cbarlabelpad=15, contour=False, norm=None,
                contour_levels=None, cmap='plasma',
                label_contours=False, contour_label_format='%1.1f',
                figsize=(10, 10), **kwargs):
        """
        Plot a 2D array with a color map and optional contours.

        :param array: they array to plot (of i,j)
        :param vmin: vmin to pass to the plot, None, number (as per matplotlib), str (e.g., 1th, 1st, 5th), the percentile to use
        :param vmax: vmax to pass to the plot, None, number (as per matplotlib), str (e.g., 1th, 1st, 5th), the percentile to use
        :param title: a title to include on the plot
        :param ax: None or a matplotlib ax to plot on top of, not frequently used (execpt in subplots)
        :param color_bar: Boolean if true plot a color bar scale
        :param base_map_path: None or a path to a raster basemap to plot under the data. if the raster is RGB then it is transformed to grayscale.
        :param cbar_lab: string to label the cbar
        :param contour: boolean if true print black contours on the map
        :param norm: None or matplotlib.colors.Normalize, if None then no normalisation is applied (other than vmin/vmax)
        :param contour_levels: see levels in matplotlib.pyplot.contour, or float/int, If float/int then contour levels are calculated from array.min()//level x level to array.max()//level x level
        :param cmap: color map to use
        :param label_contours: boolean if true then print labels in line with the contours
        :param contour_label_format: format for the contour label (see fmt in plt.clabel)
        :param figsize: figsize (ignored if ax is not None)
        :param kwargs: additional kwargs to plot to pass to pcolormesh

        :return: fig, ax
        """
        if base_map_path:
            base_map_path = Path(base_map_path)
            assert base_map_path.exists(), f'base_map_path {base_map_path} does not exist'
        default_alpha = 1 if base_map_path is None else 0.5
        alpha = kwargs.pop('alpha', default_alpha)

        assert array.shape == self.spatial_2d_shape, f'{array.shape=} must match {self.spatial_2d_shape=}'
        vmin, vmax = self._apply_vminvmax(array, vmin=vmin, vmax=vmax)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            assert isinstance(ax, plt.Axes)
            fig = ax.figure

        if base_map_path is not None:
            self._plot_basemap(ax, base_map_path)

        model_xs_edge, model_ys_edge = self._get_grid_x_y(cell_centers=False)

        if alpha == 1:
            linewidth = None
            edgecolors = 'face'
        else:
            edgecolors = 'face'
            linewidth = 0

        edgecolors = kwargs.pop('edgecolors', edgecolors)
        linewidth = kwargs.pop('linewidth', linewidth)

        pcm = ax.pcolormesh(model_xs_edge, model_ys_edge, np.ma.masked_invalid(array),
                            cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, edgecolors=edgecolors,
                            linewidth=linewidth, norm=norm, antialiased=True, **kwargs)
        contour_norm = pcm.norm
        if color_bar:
            cbar = fig.colorbar(pcm, ax=ax, extend='max', alpha=1)
            cbar.solids.set(alpha=1)
            if cbar_lab is not None:
                cbar.ax.set_ylabel(cbar_lab, rotation=270,
                                   labelpad=cbarlabelpad)

        if contour:
            model_xs, model_ys = self._get_grid_x_y(cell_centers=True)
            contour_levels = self._get_contour_levels(array, contour_levels, norm=contour_norm)

            cs = ax.contour(model_xs, model_ys, array, levels=contour_levels,
                            colors='k')
            if label_contours:
                ax.clabel(cs, inline=1, fontsize=10, fmt=contour_label_format)

        if title is not None:
            ax.set_title(title)

        xmin, xmax, ymin, ymax = self.get_xlim_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')

        return fig, ax

    def _plot_basemap(self, ax, base_map_path):
        ds = gdal.Open(str(base_map_path))
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

        image = ds.ReadAsArray()
        if image.ndim == 3:  # if a rgb image then plot as greyscale
            image = image.mean(axis=0)
            image = image.astype(float)
            image[image == 0] = np.nan
        ll = (minx, miny)
        ur = (maxx, maxy)

        ax.imshow(image, extent=[ll[0], ur[0], ll[1], ur[1]], cmap='gray', vmin=0, vmax=255)
        ds = None

        model_xs_edge, model_ys_edge = self._get_grid_x_y(cell_centers=False)
        no_flow = self.get_active_index()
        plt_nf = self.get_2d_spatial_zero(float)
        plt_nf[self.get_active_index()] = np.nan

        cmap, norm = from_levels_and_colors([0, 1], ['cyan', 'black', 'white'], extend='both')
        bk_edgecolors = 'face'
        bk_linewidth = 0
        pcm = ax.pcolormesh(model_xs_edge, model_ys_edge, np.ma.masked_invalid(no_flow), cmap=cmap, norm=norm,
                            alpha=0.5, edgecolors=bk_edgecolors, linewidth=bk_linewidth,
                            antialiased=True)

    def shapefile_to_spatial_2d(self, shp_path, attribute, alltouched=True, check_crs=True):

        if check_crs:
            self.check_shape_crs(shp_path)

        x_min, x_max, y_min, y_max = self.get_xlim_ylim()
        rows, cols = self.spatial_2d_shape
        source_ds = ogr.Open(str(shp_path))
        source_layer = source_ds.GetLayer()

        out = self._burn_layer(cols, rows, self.grid_space, self.grid_space, x_min, y_min, source_layer, alltouched,
                               shp_path,
                               attribute)
        return out

    def _burn_layer(self, cols, rows, pixelWidth, pixelHeight, x_min, y_min, use_layer, alltouched, path, attr):
        target_ds = gdal.GetDriverByName('MEM').Create('',
                                                       cols, rows,
                                                       1,
                                                       gdal.GDT_Float64)
        target_ds.SetGeoTransform((x_min, pixelWidth, 0, y_min, 0, pixelHeight))
        band = target_ds.GetRasterBand(1)
        NoData_value = -999999
        band.Fill(NoData_value)
        band.SetNoDataValue(NoData_value)
        band.FlushCache()
        if alltouched:
            opt = ["ALL_TOUCHED=TRUE", 'a_nodata={}'.format(NoData_value),
                   f"ATTRIBUTE={attr}", 'initValues={}'.format(NoData_value)]  # value set in temp layer file
        else:
            opt = ['a_nodata={}'.format(NoData_value), f"ATTRIBUTE={attr}",  # value set in temp layer file
                   'initValues={}'.format(NoData_value)]
        gdal.RasterizeLayer(target_ds, [1], use_layer, options=opt)
        target_dsSRS = osr.SpatialReference()
        target_dsSRS.ImportFromEPSG(self.epsg_num)
        target_ds.SetProjection(target_dsSRS.ExportToWkt())

        outdata = target_ds.ReadAsArray()
        outdata[np.isclose(outdata, -999999)] = np.nan
        if 0 in outdata:
            warnings.warn(
                f'0 value in the burned array from {path} this may be an actual value or may be a null value as shapefiles '
                'store null values as 0 for integer fileds')

        target_ds = None
        final_ds = None
        return np.flipud(outdata)

    def _apply_vminvmax(self, data, vmin, vmax):
        """
        allow the passing of percentiles for vmin and vmax

        :param data: data to apply vmin and vmax to (np.ndarray)
        :param vmin: None, int or string ('1th', '99th', etc)
        :param vmax: None, int or string ('1th', '99th', etc)
        """
        temp = data.copy()

        temp[~self.get_active_index()] = np.nan

        if vmin is None:
            vmin_out = np.nanmin(temp)
        elif pd.api.types.is_number(vmin):
            vmin_out = vmin
        else:
            assert isinstance(vmin, str), 'vmin must be None, number, or string'
            vmin = vmin.strip('stndrdth')  # remove st, nd rd, th
            vmin = int(vmin)
            assert vmin < 100, 'vmin must be less than 100, when specifying percentiles'
            assert vmin > 0, 'vmax must be greater than 0, when specifying percentiles'
            vmin_out = np.nanpercentile(temp, vmin)

        if vmax is None:
            vmax_out = np.nanmax(temp)
        elif pd.api.types.is_number(vmax):
            vmax_out = vmax
        else:
            assert isinstance(vmax, str), 'vmax must be None, number, or string'
            vmax = vmax.strip('stndrdth')  # remove st, nd rd, th
            vmax = int(vmax)
            assert vmax < 100, 'vmax must be less than 100, when specifying percentiles'
            assert vmax > 0, 'vmax must be greater than 0, when specifying percentiles'
            vmax_out = np.nanpercentile(temp, vmax)

        return vmin_out, vmax_out

    @staticmethod
    def _get_contour_levels(array, contour_levels, norm=None):
        """
        get the contour levels for a given array and contour levels (e.g. pass int/float for step size)

        :param array: array
        :param contour_levels: int / float for stepsize, n{int} for n levels (np.histogram_bin_edges), or array of levels
        :param norm: matplotlib norm to apply to the array when calling nbins
        :return:
        """
        if isinstance(contour_levels, str):
            assert contour_levels[0] == 'n', 'contour levels must be n{int} for any string to define nlevels'
            n = int(contour_levels[1:])
            if norm is None:
                contour_levels = np.histogram_bin_edges(array, bins=n)
            else:
                contour_levels = norm.inverse(np.histogram_bin_edges(norm(array), bins=n))

        elif contour_levels is not None and not hasattr(contour_levels, '__len__'):
            # contour levels is step size, normalisation is not applied
            start = np.nanmin(array) // contour_levels * contour_levels
            stop = np.nanmax(array) // contour_levels * contour_levels
            contour_levels = np.arange(start, stop + contour_levels, contour_levels)
        else:
            pass  # user passed contour levels
        return contour_levels

    @staticmethod
    def show():
        """
        a shortcut to be able to show plots without importing matplotlib.pyplot

        :return:
        """
        plt.show()

    @staticmethod
    def close(fig=None):
        """
        a shortcut to be able to close plots without importing matplotlib.pyplot

        :param fig: see matplotlib.pyplot.close()
        :return:
        """
        plt.close(fig)


class CompressedSpatialAccessor(_BaseAccessor):
    """
    This supports easy access to compressed spatial datasets.  That is data which has n dimensions with only 1 dimension of space.  For example, this could be a 2d Dataset of time, space.  Where the "space" dimension has point values (e.g. unique x,y).  This is opposed to an uncompressed spatial dataset which would have n dimensions with 2 dimensions of space (x, y)
    """

    def spatial_1d_to_spatial_2d(self, array, missing_value=np.nan):
        """
        convert a 1d (collapsed) spatial array to a 2d spatial array

        :param array:  1d array to convert
        :param missing_value: value to use for missing values (to support integer arrays)
        :return: array (self.spatial_2d_shape)
        """
        out = self.get_2d_spatial_zero(array.dtype) + missing_value

        try:
            np.array([missing_value]).astype(array.dtype)
        except ValueError:
            raise ValueError(
                f'missing_value {missing_value} cannot be converted to {array.dtype}, set a different missing_value')

        active_index = self.get_active_index()
        assert array.ndim == 1, f'array must be 1d, got {array.ndim}'
        assert array.shape[
                   0] == active_index.sum(), f'array shape {array.shape} does not match active index {active_index.sum()}'
        out[active_index] = array
        return out

    def spatial_2d_to_spatial_1d(self, array):
        """
        convert a 2d spatial array to a 1d (collapsed) spatial array

        :param array: 2d array to convert
        :return: array (1d)
        """
        assert array.shape == self.spatial_2d_shape, f'{array.shape=} must match {self.spatial_2d_shape=}'
        active_index = self.get_active_index()
        return array[active_index]


    def get_closest_loc_to_point(self, nztmx, nztmy, coords_out_domain='raise'):
        """
        get the closest spatial index(s) to a point in the spatial domain

        :param nztmx: single or array of x coordinates
        :param nztmy: single or array of y coordinates
        :param coords_out_domain: ['raise', 'coerce' or 'pass'].  What to do if the coordinates are outside the domain

        - 'raise': raise a ValueError
        - 'coerce': return -1 for out of domain coords (note that this may still be a valid index, so care must be taken)
        - 'pass': returns the closest index, but this may be WELL outside the domain


        :return: index(s) of the closest point(s). an integer if single point, or an array of integers if multiple points

                - note that this is the index of the spatial points (which does not include inactivate cells)
        """
        return_single = False
        if pd.api.types.is_number(nztmx):
            return_single = True

        nztmx = np.atleast_1d(nztmx)
        nztmy = np.atleast_1d(nztmy)
        assert nztmx.shape == nztmy.shape, f'{nztmx.shape=} must match {nztmy.shape=}'

        xmin, xmax, ymin, ymax = self.get_xlim_ylim()

        out_of_bounds = ((nztmx < xmin) | (nztmx > xmax) | (nztmy < ymin) | (nztmy > ymax))
        if coords_out_domain == 'raise' and out_of_bounds.any():
            raise ValueError(f'coordinates at indexes {np.where(out_of_bounds)} are out of domain')

        if coords_out_domain == 'coerce' and out_of_bounds.any():
            return -1

        xs, ys = self._get_grid_x_y(cell_centers=True)
        xs = xs.flatten()
        ys = ys.flatten()
        active_index = self.get_active_index().flatten()
        xs = xs[active_index]
        ys = ys[active_index]

        nztmx = np.full((len(nztmx), len(xs)), nztmx)
        nztmy = np.full((len(nztmy), len(xs)), nztmy)

        dist_x = nztmx - xs
        dist_y = nztmy - ys
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        loc = np.nanargmin(dist, axis=1)

        if return_single:
            return loc[0]
        else:
            return loc








