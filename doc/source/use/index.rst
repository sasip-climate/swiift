**********
User guide
**********

|project| was developed with the goal of studying the impact of ocean waves on
sea ice, through their shaping of the floe size distribution.

Sea ice forms in polar regions when seawater freezes, and an ice floe is a
piece of floating sea ice.
The floe size distribution is a statistical way to describe the sizes of a
collection of floes, without having to describe each floe individually.
The floe size distribution changes under the action of a collection of factors,
including the lateral melt and growth of the ice, the rafting of existing
floes, or the fracture of individual floes into smaller floes by the ice
internal stress, ocean currents, or the bending stress imposed by waves.
|project| focuses on the latter.

We choose to model this phenomenon in isolation, which implies making
simplifying assumptions.


Quickstart
==========

|project| can be installed from PyPI:

.. code:: sh

    pip install SWIIFT

It can also be installed from GitHub:

.. code:: sh

   pip install git+ssh://git@github.com/sasip-climate/swiift.git@$ver#egg=swiift

with :code:`$ver` an optional release tag.

We strongly recommend installing |project| in a dedicated virtual environment.


Overview
========

|project| was designed to build time-domain simulations of wave fracture.
It allows some flexibility in how these simulations are constructed.
To achieve this, we rely on a collection of components.
Physical objects---such as ice floes---or concepts---such as a fracture
mechanism---are abstracted into objects of varying complexity, which can then
be combined into higher-level objects.

#. :py:class:`~swiift.Experiment` -- the user-facing interface to simulations.
   It combines a :py:class:`~swiift.model.model.Domain`, a fracture Handler,
   and add a time dimension.
   It exposes convenience methods to step through a simulation, save its
   results, or load existing results from disk.
#. :py:class:`~swiift.model.model.Domain` -- the state of an experiment at a given time.
   It holds informations on the state of the ice (that is, a collection of
   :py:class:`~swiift.model.model.WavesUnderFloe`), the fluid that bears it (a
   :py:class:`~swiift.model.model.FreeSurfaceWaves` object), and forcing waves
   (:py:class:`~swiift.DiscreteSpectrum`).
#. Fracture handler -- the parametrisation of fracture.
   It determines whether floes in a domain should break.
#. :class:`~swiift.DiscreteSpectrum` -- |project| handles monochromatic or
   polychromatic sine forcings.
   A polychromatic is a linear superposition of wave modes.
   These objects gather amplitudes, frequencies, and phases of these wave
   modes.
   These can be automatically built from spectral parametrisations exposed by
   :py:mod:`swiift.api.spectra`; attenuation is handled at the domain level.
   It might be moved to the floe level in the future, to allow for more
   flexibility.
#. :class:`~swiift.Floe` -- from the point of view of |project|, floes are
   segments, localised in space, that carry information on ice properties
   (fixed for a given floe).
   When part of a domain, they are promoted to
   :class:`~swiift.model.model.WavesUnderFloe`, and get knowledge of the wave
   state (that varies along the floe).
   The latter have associated methods to compute physical quantities deriving
   from these properties and the wave forcing, such as the the vertical
   displacement or the elastic potential energy.
#. Wave attenuation -- the presence of floes may attenuate wave propagation.
   The module :py:mod:`swiift.lib.att` exposes builtin attenuation schemes,
   and users can define their own schemes as well.


Examples
========

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Basic experiment setup <../notebooks/Experiment.md>
