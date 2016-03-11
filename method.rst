.. _method:

=================
Behind the Scenes
=================

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
:math:`x^*` for which:

.. math::
   :nowrap:

   \begin{eqnarray}
      f(x^*) &>& \epsilon_0 \\
      \left.\frac{{\rm d}^2f}{{\rm d}x^2}\right|_{x=x^*}  & < & 0 \\
      \left.\frac{{\rm d}^3f}{{\rm d}x^3}\right|_{x=x^*}  & = & 0 \\
      \left.\frac{{\rm d}^4f}{{\rm d}x^4}\right|_{x=x^*}  & > & 0.
   \end{eqnarray}
   :label: constraints

The reason for selecting these constraints is as follows:

* The inequality :eq:`constraints`

* F
