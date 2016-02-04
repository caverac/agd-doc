.. _tutorial:

===================
Quickstart Tutorial
===================
  



After creating the data, we will train the smoothing parameter alpha to the
optimal value.

.. code-block:: python

    import gausspy.GaussianDecomposer as gp

    g = gp.GaussianDecomposer()

    # Load the training data from the pickle file
    g.load_training_data('agd_data.pickle')

    # One phase training
    g.set('phase', 'one')

    # threshold below which Gaussian components will not be fit
    g.set('SNR_thresh', 5.)

    # Find the optimal alpha value with the training dataset given an initial
    # guess for the alpha value 
    g.train(alpha1_initial = 10.)

Now we can see the trained value of alpha by looking at the attribute of our
GaussianDecomposer instance.

.. code-block:: python

    # get the parameters attribute of g, which is a dictionary of important
    # variables
    print(g.p['alpha1'])






