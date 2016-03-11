.. _tutorial:

=======================================
Simple Example Tutorial
=======================================

Constructing a GaussPy-Friendly Dataset
--------------------------------------
Before implementing AGD, we first must put data into a format readable by GaussPy. GaussPy requires the indepenent and dependent spectral arrays (e.g., channels and amplitude) and an estimate of the per-channel noise in the specrum.

To begin, we can create a simple Gaussian function of the form:

.. math::
    S(x_i) = \sum_{k=1}^{\texttt{NCOMPS}} {\texttt{AMP}_k} \exp\left[-\frac{4\ln 2 (x_i
    - \texttt{MEAN}_k)^2}{\texttt{FWHM}_k^2} \right] + \texttt{NOISE},
    \qquad i = 1, \cdots, \texttt{NCHANNELS}
    :label: spectra

where,

1. ``NCOMPS`` is the number of Gaussian components in each spectrum

2. ``(AMP, MEAN, FWHM)`` are the parameters of each Gaussian component

3. ``NCHANNELS`` is the number of channels in the spectrum (sets the resolution)

4. ``NOISE`` is the level of noise introduced in each spectrum, described by the root mean square (RMS) noise per channel.

In the next example we will show how to implement this in python. We
have made the following assumptions:

1. ``NCOMPS = 1`` (to begin with a simple, single Gaussian)

2. ``AMP = 1.0, MEAN = 256, FWHM = 20`` (fixed Gaussian parameters)

3. ``NCHANNELS = 512``

4. ``RMS = 0.05``

The following code describes an example of how to create spectrum with Gaussian shape and store the channels, amplitude and error arrays in a python pickle file to be read later by GaussPy.

.. literalinclude:: simple_gaussian.py
    :language: python
    :lines: 1-34

.. plot:: simple_gaussian.py

Running GaussPy
----------------------------
With our simple dataset in hand, we can use GaussPy to decompose the spectrum into Gaussian functions. To do this, we must specify the smoothing parameter :math:`\alpha`. For now, we will guess a value of :math:`\alpha=10`. Later in this chapter we will learn about training AGD to select the optimal value of :math:`\alpha`.

The following is an example code for running GaussPy. We will use the "one-phase" decomposition to begin with. We must specify the following parameters:

1. ``alpha1``: our choice for the value of :math:`\alpha`.

2. ``snr_thresh``: the signal-to-noise ratio threshold below which amplitude GaussPy will not fit a component.

3. ``DATA``: the filename containing the dataset to-be-decomposed, constructed in the previous section (or any GaussPy-friendly dataset)

4. ``DATA_out``: filename to store the decomposition results from GaussPy.

.. code-block:: python

    # Decompose simple dataset using AGD
    import pickle
    import gausspy.gp as gp

    # Specify necessary parameters
    alpha1 = 10.
    snr_thresh = 5.
    DATA = 'simple_gaussian.pickle'
    DATA_out = 'simple_gaussian_decomposed.pickle'

    # Load GaussPy
    g = gp.GaussianDecomposer()

    # Setting AGD parameters
    g.set('phase', 'one')
    g.set('SNR_thresh', [snr_thresh, snr_thresh])
    g.set('alpha1', alpha1)
    g.set('mode','conv')

    # Run GaussPy
    decomposed_data = g.batch_decomposition(DATA)

    # Save decomposition information
    pickle.dump(decomposed_data, open(DATA_out, 'w'))

After AGD determines the Gaussian decomposition, GaussPy then performs a least squares fit of the inital AGD model to the data to produce a final fit solution. The file containing the fit results is a python pickle file. The contents of this file can be viewed by printing the keys within the saved dictionary via,

.. code-block:: python

    print decomposed_data.keys()

The most salient information included in this file are the values for the ``amplitudes``, ``fwhms`` and ``means`` of each fitted Gaussian component. These include,

1. ``amplitudes_initial, fwhms_initial, means_initial`` : the parameters of each Gaussian component determined by AGD (each array has length equal to the number of fitted components).

2. ``amplitudes_fit, fwhms_fit, means_fit`` : the parameters of each Gaussian component following a least-squares fit of the initial AGD model to the data.

3. ``amplitudes_fit_err, fwhms_fit_err, means_fit_err`` : uncertainities in the fitted Gaussian parameters, determined from the least-squares fit.

GaussPy also stores the reduced :math:`\chi^2` value from the least-squares fit (``rchi2``), but this is currently under construction. This value can be computed outside of GaussPy easily.


Plot Decomposition Results
----------------------------

The following is an example python script for plotting the original spectrum and GaussPy decomposition results. We must specify the following parameters:

1. ``DATA``: the filename containing the dataset to-be-decomposed.

2. ``DATA_decomposed``: the filename containing the GaussPy decomposition results.

.. literalinclude:: simple_gaussian_plot.py
    :language: python

.. plot:: simple_gaussian_plot.py

Clearly the fit to the simple Gaussian spectrum is good. If we were to vary the value of :math:`\alpha`, the fit would not change significantly as the fit to a spectrum containing a single Gaussian funciton does not depend sensitively on the initial guesses, especially because GaussPy performs a least-squares fit after determining initial guesses for the fitted Gaussian parameters with AGD.

We can now move on from the simple example above to vary the complexity of the spectra to be decomposed, as well as the effect of different values of :math:`\alpha` on the decomposition.

=============================
Multiple Gaussians Example
=============================


Constructing a GaussPy-Friendly Dataset
--------------------------------------
As discussed in the Simple Example section above, before running GaussPy we must ensure that our data is in a format readable by GaussPy. In particular, for each spectrum, we need to provide the independent and dependent spectral arrays (i.e. channels and amplitudes) and an estimate of the uncertainity per channel. In the following example we will construct a spectrum containing multiple overlapping Gaussian components with added spectral noise, using Equation :eq:`spectra`, and plot the results.

We will make the following choices for parameters in this example:

1. ``NCOMPS = 3`` : to include 3 Gaussian functions in the spectrum

2. ``AMPS = [3,2,1]`` : amplitudes of the included Gaussian functions

3. ``FWHMS = [10,20,30]`` : FWHM (in channels) of the included Gaussian functions

4. ``MEANS = [10,20,30]`` : mean positions (in channels) of the included Gaussian functions

5. ``NCHANNELS = 512`` : number of channels in the spectrum

6. ``RMS = 0.05`` : RMS noise per channel

The following code provides an example of how to construct a Gaussian funtion with the above parameters and store it in GaussPy-friendly format.

.. literalinclude:: multiple_gaussians.py
    :language: python
    :lines: 1-41

A plot of the spectrum constructed above is included below.

.. plot:: multiple_gaussians.py

Running GaussPy
----------------
With our GaussPy-friendly dataset, we can now run GaussPy. As in the simple example (Chaper 3), we begin by selecting a value :math:`\alpha=` to use in the decomposition. In this case we will select :math:`\alpha=20`. As before, the important parameters to specify are:

1. ``alpha1``: our choice for the value of :math:`\alpha`.

2. ``snr_thresh``: the signal-to-noise ratio threshold below which amplitude GaussPy will not fit a component.

3. ``DATA``: the filename containing the dataset to-be-decomposed, constructed above (or any GaussPy-friendly dataset)

4. ``DATA_out``: filename to store the decomposition results from GaussPy.

.. literalinclude:: multiple_gaussians_plot.py
    :language: python
    :lines: 1-25

Plot Decomposition Results
----------------------------

Following the decomposition by GaussPy, we can explore the effect of the choice of :math:`\alpha` on the decomposition. Below, we have run GaussPy on the multiple-Gaussian dataset constructed above for three values of :math:`\alpha`, including :math:`\alpha=20, \alpha = 2` and :math:`\alpha=10`.

.. plot:: multiple_gaussians_plot.py

These results demonstrate that our choice of :math:`\alpha` has a significant effect on the success of the GaussPy model. In order to select the right value of :math:`\alpha` for a given dataset, we need to train the AGD algorithm using a training set. This process is described in the following section.


=============================
Training AGD to select Alpha
=============================

Creating a Synthetic Training Dataset
----------------------------

To select the optimal value of the smoothing parameter :math:`\alpha`, you must train the AGD algorithm using a training dataset with known underlying Gaussian decomposition. In other words, you need to have a dataset for which you know (or have an estimate of) the true Gaussian model. This training dataset can be composed of real (i.e. previously analyzed) or synthetically-constructed data, for which you have prior information about the underlying decomposition. This prior information is used to maximize the model accuracy by calibrating the :math:`\alpha` parameter used by AGD.

Training datasets can be constructed by adding Gaussian functions with parameters drawn from known distributions with known uncertainties. For example, we can create a mock dataset with ``NSPECTRA``-realizations of Equation :eq:`spectra`.

In the next example we will show how to implement this in python. We have made the following assumptions

1. :math:`\mathrm{NOISE} \sim N(0, {\rm RMS}) + f \times {\rm RMS}`
   with ``RMS=0.05`` and :math:`f=0`

2. ``NCOMPS = 4``

3. ``NCHANNELS = 512`` This number sets the resolution of each
   spectrum. **Does this number need to be the same for all spectra in
   AGD?**

4. :math:`\mathrm{AMP} \sim \mu(5 \mathrm{RMS}, 25 \mathrm{RMS})`,
   this way we ensure that every spectral feature is above the noise
   level. Spectra with a more dominant contribution from the noise can
   also be generated and used as training sets for AGD

5. :math:`\mathrm{FWHM} \sim \mu(10, 35)` and :math:`\mathrm{MEAN}
   \sim \mu(0.25, 0.75) \times \mathrm{NCHANNELS}`, note that for our
   choice of the number of channels, this selection of ``FWHM``
   ensures that even the wider component can be fit within the
   spectrum.

.. code-block:: python

    # Create training dataset with Gaussian profiles

    import numpy as np
    import pickle

    # Specify the number of spectral channels (NCHANNELS)
    NCHANNELS = 512
    # Specify the number of spectra (NSPECTRA)
    NSPECTRA = 200

    # Estimate of the root-mean-square uncertainty per channel (RMS)
    RMS = 0.05

    # Estimate the mean number of Gaussian functions to add per spectrum (NCOMPS)
    NCOMPS = 4

    # Specify the min-max range of possible properties of the Gaussian function paramters:
    # Amplitude (AMP)
    AMP_lims = [RMS * 5, RMS * 25]
    # Full width at half maximum in channels (FWHM)
    FWHM_lims = [10, 35] # channels
    # Mean channel position (MEAN)
    MEAN_lims = [0.25 * NCHANNELS, 0.75 * NCHANNELS]

    # Indicate whetehre the data created here will be used as a training set
    # (a.k.a. decide to store the "true" answers or not at the end)
    TRAINING_SET = True

    # Specify the pickle file to store the results in
    FILENAME = 'agd_data_science.pickle'

With the above parameters specified, we can proceed with constructing a set of synthetic training data composed of Gaussian functions with known parameters (i.e., for which we know the "true" decompositon),sampled randomly from the parameter ranges specified above. The resulting data, including the channel values, spectral values and error estimates, are stored in the pickle file specified above. If we want this to be a training set (``TRAINING_SET = True``), the "true" decomposition answers for estimating the accuracy of a decomposition are also stored in the output file. For example, to construct a synthetic dataset:

.. code-block:: python

    # GaussPy Example 1
    # Create spectra with Gaussian profiles -cont-

    # Initialize
    agd_data = {}
    chan = np.arange(NCHANNELS)
    errors = chan * 0. + RMS # Constant noise for all spectra

    # Begin populating data
    for i in range(NSPECTRA):
        spectrum_i = np.random.randn(NCHANNELS) * RMS

        # Sample random components:
        amps = np.random.rand(NCOMPS) * (AMP_lims[1] - AMP_lims[0]) + AMP_lims[0]
        fwhms = np.random.rand(NCOMPS) * (FWHM_lims[1] - FWHM_lims[0]) + FWHM_lims[0]
        means = np.random.rand(NCOMPS) * (MEAN_lims[1] - MEAN_lims[0]) + MEAN_lims[0]

        # Create spectrum
        for a, w, m in zip(amps, fwhms, means):
            spectrum_i += gaussian(a, w, m)(chan)

        # Enter results into AGD dataset
        agd_data['data_list'] = agd_data.get('data_list', []) + [spectrum_i]
        agd_data['x_values'] = agd_data.get('x_values', []) + [chan]
        agd_data['errors'] = agd_data.get('errors', []) + [errors]

        # If training data, keep answers
        if TRAINING_SET:
            agd_data['amplitudes'] = agd_data.get('amplitudes', []) + [amps]
            agd_data['fwhms'] = agd_data.get('fwhms', []) + [fwhms]
            agd_data['means'] = agd_data.get('means', []) + [means]

    # Dump synthetic data into specified filename
    pickle.dump(agd_data, open(FILENAME, 'w'))


Training the Algorithm
----------------------------

Next, we will apply GaussPy to the real or synthetic training dataset and compare the results with the known underlying decompositon to determine the optimal value for the smoothing parameter :math:`\alpha`. We must set the following parameters

1. ``FILENAME``: the filename of the training dataset in GaussPy-friendly format.

2. ``snr_thresh``: the signal-to-noise threshold below which amplitude GaussPy will not fit components.

3. ``alpha1``: initial guess for :math:`\alpha`

.. code-block:: python

    import gausspy.gp as gp

    # Set necessary parameters
    FILENAME = 'agd_data.pickle'
    snr_thresh = 5.
    alpha1

    g = gp.GaussianDecomposer()

    # Next, load the training dataset for analysis:
    g.load_training_data(FILENAME)

    # Set GaussPy parameters
    g.set('phase', 'one')
    g.set('SNR_thresh', snr_thresh)

    # Train AGD starting with initial guess for alpha
    g.train(alpha1_initial = alpha1, plot=False,
        verbose = False, mode = 'conv',
        learning_rate = 1.0, eps = 1.0, MAD = 0.1)

GausspPy will iterate over a range of :math:`\alpha` values and compare the decomposition associated with each :math:`\alpha` value to the correct decomposition specified within the training dataset to maximize the accuracy of the decomposition.

Once the training is completed, we can view the "trained" value of alpha by looking at the attribute of our GaussianDecomposer instance.

.. code-block:: python

    # get the parameters attribute of g, which is a dictionary of important
    # variables
    print(g.p['alpha1'])

========================================
Running AGD using Trained Alpha
========================================

With the trained value of :math:`\alpha` in hand, we can proceed to decompose our target dataset with AGD. 



