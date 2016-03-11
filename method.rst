.. _method:

=================
Behind the Scenes
=================

Basic concepts
--------------

``GaussPy`` is a Python implementation of the AGD algorithm described
in `Lindner et al. (2015), AJ, 149, 138
<http://iopscience.iop.org/article/10.1088/0004-6256/149/4/138/meta>`_. At
its core, AGD is a fast, automatic, extremely versatile way of
providing initial guesses for fitting Gaussian components to a
function of the form :math:`f(x) + n(x)`, where :math:`n(x)` is a term
modeling possible contributions from noise. It is important to
emphasize here that although we use terminology coming from
radio-astronomy all the ideas upon which the code is founded can be
applied to any function of this form, moreover, non-Gaussian
components can also be in principle extracted with our methodology,
something we will include in a future release of the code.


Ideally, if blending of components was not an issue and :math:`n(x)=0`
the task of fitting Gaussians to a given spectrum would be reduced to
find local maxima of :math:`f(x)`. However, both of these assumptions
dramatically fail in practical applications, where blending of lines
is an unavoidable issue and noise is intrinsic to the process of data
acquisition. In that case, looking for solutions of :math:`{\rm
d}f(x)/{\rm d}x = 0` is not longer a viable route to find local
extrema of :math:`f(x)`, instead a different approach must be taken.

AGD uses the fact that a local maximum in :math:`f(x)` is also a local
minimum in the curvature. That is, the algorithm looks for points
:math:`x^*` for which the following conditions are satisfied.

* The function :math:`f(x)` has a non-trivial value

.. math::  f(x^*) > \epsilon_0.
   :label: f0const

In an ideal situation where the contribution from noise vanishes we
can take :math:`\epsilon_0=0`. However, when random fluctuations are
added to the target function, this condition needs to be modified
accordingly. A good selection of :math:`\epsilon_0` thus needs to be
in proportion to the RMS of the analized signal.
      

* Next we require that the function :math:`f(x)` has a "bump" in
  :math:`x^*`

.. math::  \left.\frac{{\rm d}^2f}{{\rm d}x^2}\right|_{x=x^*}  < 0,
   :label: f2const

this selection of the inequality ensures also that such feature has
negative curvature, or equivalently, that the point :math:`x^*` is
candidate for being the position of a local maximum of
:math:`f(x)`. Note however that this is not a sufficient condition, we
also need to ensure that the curvature has a minimum at this location.
      
* This is achived by imposing two additional constraints on
  :math:`f(x)`

.. math:: \left.\frac{{\rm d}^3f}{{\rm d}x^3}\right|_{x=x^*} = 0
   :label: f3const

.. math:: \left.\frac{{\rm d}^4f}{{\rm d}x^4}\right|_{x=x^*} > 0
   :label: f4const
            

These 4 constraints then ensure that the point :math:`x^*` is a local
minimum of the curvature. Furthermore, even in the presence of both
blending and noise, these expressions will yield the location of all
the points that are possible canditates for the positions of Gaussian
components in the target function. Fig. :num:`#curvature` is an
example of a function defined as the sum of three gaussians for which
the conditions Eq. :eq:`f0const` - :eq:`f4const` are satisfied and the
local minima of curvature are successfully found, even when blending
of components is relevant.

.. _curvature:

.. figure:: curvature.pdf
    :width: 4in
    :align: center
    :figclass: align-center
    :alt: alternate text

    Example of the points of negative curvature of the function
    :math:`f(x)`. In this case :math:`f(x)` is the sum of three
    independent Gaussian functions (top). The vertical lines in each
    panel show the conditions imposed on the derivatives to define the
    points :math:`x^*`.



Dealing with noise
------------------

The numeral problem related to the solution shown in the previous
section comes from the fact that calculating Eq. :eq:`f2const` -
:eq:`f4const` is not trivial in the presence of noise.
