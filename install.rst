.. _install:

===============
Installation
===============


------------
 Dependencies
------------

*  `Python 2.7 <http://www.numpy.org/>`_

* `Numpy <http://www.numpy.org/>`_

* `Scipy <http://www.scipy.org/>`_

* `h5py <http://www.h5py.org/>`_

* `GNU Scientific Library (GSL) <http://www.gnu.org/software/gsl/>`_


If you do not already have Python 2.7, you can
install the 
`Anaconda Scientific Python distribution <https://store.continuum.io/cshop/anaconda/>`_, 
which comes pre-loaded with Numpy, Scipy, and h5py.

To obtain GSL:

.. code-block:: none

   sudo apt-get install libgsl0-dev


---------------
Download GaussPy
---------------

Download GaussPy from...

-----------------------
Installing Dependencies
-----------------------

You will need several libraries which the GSL, h5py, and scipy libraries depend
on. 

$ sudo apt-get install libblas-dev liblapack-dev gfortran libgsl0-dev
libhdf5-serial-dev hdf5-tools -y

Install pip for easy installation of python packages:

$ sudo apt-get install python-pip
$ sudo pip install --upgrade pip

Install `scipy`:

$ sudo pip install -I scipy==0.17.0

Install `lmfit`

$ sudo pip install lmfit

Install `numpy`

$ sudo apt-get install python-numpy -y

Install `matplotlib`

$ sudo apt-get install -qq python-matplotlib -y

Install `h5py`

$ sudo apt-get install -qq python-h5py -y

------------
Installing GaussPy
------------

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. Then cd to the local
directory containing gausspy and type

$ python setup.py install

If you don't have root access and/or wish a local installation of
gausspy then use

$ python setup.py install --user

change the 'requires' statement in setup.py to include scipy and lmfit

