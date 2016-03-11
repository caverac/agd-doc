.. _method:

=================
Behind the Scenes
=================

At its core GaussPy uses numerical derivatives to find interesting
points in a given spectrum. It is important to emphasize here that we
use terminology coming from radio-astronomy but all the ideas upon
which the code is developed can be applied to any function of the form
:math:`f(x) + n(x)`, where :math:`n(x)` is a term modeling possible
contributions from noise.

In Figure 
