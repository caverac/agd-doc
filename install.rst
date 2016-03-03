.. _install:

===============
Installation
===============


------------
Dependencies
------------

*  `python 2.7 <http://www.numpy.org/>`_

* `numpy (\> v1.6.1) <http://www.numpy.org/>`_

* `scipy (\> v0.17.0) <http://www.scipy.org/>`_

* `lmfit (\> v0.9.2) <https://lmfit.github.io/lmfit-py/intro.html>`_

* `h5py (\> v2.0.1) <http://www.h5py.org/>`_


If you do not already have Python 2.7, you can
install the 
`Anaconda Scientific Python distribution <https://store.continuum.io/cshop/anaconda/>`_, 
which comes pre-loaded with `numpy`, `scipy`, and `h5py`.

---------------------
Optional Dependencies
---------------------

* `matplotlib (\> v1.1.1) <http://matplotlib.org/>`_

* `GNU Scientific Library (GSL) <http://www.gnu.org/software/gsl/>`_

---------------
Download GaussPy
---------------

Download GaussPy from...

--------------------------------
Installing Dependencies on Linux
--------------------------------

You will need several libraries which the GSL, h5py, and scipy libraries depend
on. Install these required packages with:

.. code-block:: bash

    sudo apt-get install libblas-dev liblapack-dev gfortran libgsl0-dev libhdf5-serial-dev hdf5-tools

Install pip for easy installation of python packages:

.. code-block:: bash

    sudo apt-get install python-pip

Then install the required python packages:

.. code-block:: bash

    sudo pip install scipy numpy h5py lmfit

Install the optional dependencies for plotting and optimization:

.. code-block:: bash

    sudo pip install matplotlib
    sudo apt-get install libgsl0-dev

------------------
Installing GaussPy
------------------

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. Then cd to the local
directory containing gausspy and type

$ python setup.py install

If you don't have root access and/or wish a local installation of
gausspy then use

$ python setup.py install --user

change the 'requires' statement in setup.py to include scipy and lmfit

