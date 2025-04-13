Kо̄manawa-|light_name|
#########################################

:Author: |author|
:copyright: |copyright|
:Version: |release|
:Date: |today|


.. toctree::
    :maxdepth: 2
    :hidden:

    Code documentation<autoapi/komanawa/easy_nc_accessor/index.rst>

Overview
===========

This package is designed to make it easier to access and manipulate NetCDF4 datasets provided by Kо̄manawa Solutions Ltd. within geospatial contexts.  The datasets must be in a specific format to use this package, but if you need to work with some third party data... the code within this package is open source and you are welcome to use it as a reference or to modify it to suit your needs, just note that if you use this package, you are bound by the terms of the GNU Lesser General Public License v3.0 (LGPL-3.0) license --> Essentially you may use this package, but if you **modify and redistribute** it, you must make the redistributed package open source and provide it under the same license.

Installation
=====================

Python environment
--------------------------

This package uses GDAL, which can be a real pain to manage appropriately without conda.
Therefore, we recommend using a conda environment for this package as follows:

.. code-block:: bash

    Conda create -c conda-forge --name myenv python numpy=1.26 pandas matplotlib netcdf4 pyproj geopandas gdal

    conda activate myenv

Install the package using pip:
-------------------------------------

The easiest way to install the package is to use pip:

.. code-block:: bash

    pip install git+https://github.com/Komanawa/komanawa-easy-nc-accessor.git


For a specific version of the package:


.. code-block:: bash

    pip install git+https://github.com/Komanawa/komanawa-easy-nc-accessor.git@{version}

    # example:
    pip install git+https://github.com/Komanawa/komanawa-easy-nc-accessor.git@v0.0.1

Bleeding edge version:

.. code-block:: base

    pip install git+https://github.com/Komanawa/komanawa-easy-nc-accessor.git@development


Dataset Types
=====================

This package is designed to work with specific types of datasets provided by Kо̄manawa Solutions Ltd.

Compressed Spatial datasets
-------------------------------------

This dataset type is a NetCDF4 dataset with the spatial dimensions compressed into a single dimension.
This is done as a way to reduce the working memory size of the data where large portions of the N-S / E-W oriented regular grid are not used.

This dataset is typically noted as having a "space" dimension rather than a "x" and "y" dimension.
Note the dataset will contain the following useful variables:

* `grid_x`: a 1d array that contains the x coordinates of the spatial indexes on a regular N-S, E-W grid (dimension = "grid_x").
* `grid_y`: a 1d array that contains the y coordinates of the spatial indexes on a regular N-S, E-W grid (dimension = "grid_y").
* `x`: a 1d array that contains the x coordinates of the data on the compressed grid (dimension = "space").
* `y`: a 1d array that contains the y coordinates of the data on the compressed grid (dimension = "space").
* `active_index`: a 2d boolean array that indicates which spatial indexes are active (True) and which are not (False), (dimension = ("grid_x", "grid_y")).


Usage Overview
=======================

Brush up on Numpy indexing, slicing and along axis operations
--------------------------------------------------------------------

It is beyond the scope of this documentation to fully cover the use of Numpy, but understanding indexing, slicing and array operations are essential to use multidimensional data in Python.

If you need a refresher, I personally recommend https://python-course.eu/numerical-programming/introduction-to-numpy.php

Very basically Numpy arrays are n-dimensional arrays.
It is often easiest to think of the dimensions by their meaning rather than trying to visualise the array.
Visualisation makes sense with 1, 2, or 3 dimensions, but beyond that it can break your brain.

For example consider an array with shape (100, 10, 5, 5) (axis=0, axis=1, axis=2, axis=3) which corresponds to (time, site, depth, realisation).
This is really difficult to visualise, but if you think of it as a 4D array with 100 time steps, 10 sites, 5 depths and 5 realisations, it is much easier to understand.

Numpy indexing is zero based, so the first element of an array is at index 0.
So if we want the first time step, 3rd site, 2nd depth and 4th realisation, we would use the following code:

.. code-block:: python

    import numpy as np

    # create a random array with shape (100, 10, 5, 5)
    np.random.seed(0)  # for reproducibility
    data = np.random.rand(100, 10, 5, 5).round(3)

    # get the first time step, 3rd site, 2nd depth and 4th realisation
    value = data[0, 2, 1, 3]

    print(value)  # should get 0.466

Slicing is a way to get a number of elements from an array, so for example if we want the first 5  time steps, 3rd site, 2nd depth and all realisations, we would use the following code:

.. code-block:: python

    import numpy as np

    # create a random array with shape (100, 10, 5, 5)
    np.random.seed(0)  # for reproducibility
    data = np.random.rand(100, 10, 5, 5).round(3)
    # get all time steps, 3rd site, 2nd depth and all realisations
    value = data[:5, 2, 1, :]

    print(value.shape)  # should get (5, 5)
    print(value)  # should get a 5x5 array
    # [[0.161 0.653 0.253 0.466 0.244]
    #  [0.232 0.132 0.053 0.726 0.011]
    #  [0.091 0.228 0.41  0.623 0.887]
    #  [0.709 0.954 0.352 0.898 0.77 ]
    #  [0.233 0.311 0.791 0.715 0.558]]

Axis operations are a way to perform operations along a specific axis of an array.
For example if we want to get the mean of all time steps, 3rd site, 2nd depth and all realisations, we would use the following code:

.. code-block:: python

    import numpy as np

    # create a random array with shape (100, 10, 5, 5)
    np.random.seed(0)  # for reproducibility
    data = np.random.rand(100, 10, 5, 5).round(3)
    # get the mean of all time steps, 3rd site, 2nd depth and all realisations
    value = data[:, 2, 1, :].mean(axis=0)

    print(value.shape)  # should get (5,)
    print(value)  # should get a 1D array with the mean of each realisation
    # [0.47146 0.52169 0.46669 0.49182 0.52673]


Accessing data via the NetCDF4 python package
------------------------------------------------

A full explanation of the NetCDF4 package is beyond the scope of this documentation, but a good starting point is the official documentation at: https://unidata.github.io/netcdf4-python/

Basically, the NetCDF4 package stores self documenting N-dimensional data in a binary format.


The dataset has Dimensions, Variables, and attributes that describe the data, such as the units, long name, and other metadata (for both the full dataset and each variable).

As an example .. todo make dummy dataset for use in examples  # show data, variables, dimensions


You can think of the NetCDF4 dataset as a fancy Numpy array with a lot of extra information.
You can get the dataset just like you would for a Numpy array. For example:  .. todo make dumy dataset, show read all data, slicing, integer indexing, etc.

Using this package
------------------------------------------------

Given the utility of Numpy and NetCDF4, you might reasonably wonder why this package exists.
It is to make it easier to access and manipulate, and interrogate the data with geospatial contexts.

For example this package makes it easy to:

#. Take a NetCDF4 dataset that you have loaded and transformed into just the spatial dimension and then
    #. export it as a GeoTIFF.
    #. plot a heat map of the data.
#. Find the nearest spatial indexes to a give point.
#. Create a boolean spatial index for all data that intersects a given polygon.
#. Find the correct files for a tiled dataset from a bounding box.

Rather than try to explain all the features of this package, we have provided a number of examples below and have generated documentation for the code itself.

Examples
=======================

We have provided a number of examples to show how to use the package.
If you want to run these examples locally you will need to **clone the repository (not pip install)** and then run the examples from the `examples` directory.

The examples include:

* annual_mean_at_point_polygon.py: This example shows how to get the annual mean of a datasets at a point, the total annual mean of the dataset within a polygon, and how to plot and export the results as a GeoTIFF.
* data_in_polygon.py: This example shows how to export the time series data within a polygon as a CSV file.
* export_data_at_point.py: This example shows how to export the time series data at a point as a CSV file.
* get_relevant_tiles.py:
* mean_uncert_at_point.py: This example shows how to get the mean and uncertainty of a dataset at a point, and how to plot and export the results as a GeoTIFF.
* plot_export_spatial_mean.py: This example shows hot ot plot the spatial mean of a dataset and export it as a GeoTIFF.

Example dataset
----------------------

We have provided a dummy example dataset in the `examples` directory: .. todo make dataset and put file name here

It is a NetCDF4 dataset with the following dimensions:

* time: daily data from 2020-01-01 - 2023-12-31 1461 time steps
* site: 10 sites .. todo more/less?
* uncertainty: 5 realisations that provide the uncertainty of the data

Please note that this data is roughly based on ERA5-Land Precipitation data, but should not be used for any real analysis.

.. todo add a topomap to the repo.

Bugs, issues and feature requests
=====================================

We are trying to strike the right balance between making this package easy to use and still flexible enough to be useful. If you have and issues, bugs or feature requests, please let us know by creating an issue on the GitHub repository and/or contacting us directly.