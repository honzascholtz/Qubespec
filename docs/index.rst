QubeSpec
========

QubeSpec is a simple but powerful python package to fit optical astronomical spectra and more importantly analysing IFS cube from JWST/NIRSpec, JWST/MIRI, VLT/KMOS and VLT/SINFONI. The code has built models for fitting Halpha, [OIII], Hbeta, [SII] and [NII] of galaxies, galaxies with outflows, Type-1 AGN and Quasars.

Authors: Jan Scholtz, Francesco D'Eugenio and Ignas Juodžbalis


What can QubeSpec do?
---------------------

QubeSpec takes you from a raw IFS data cube to science-ready emission-line maps:

* **Load cubes** from JWST/NIRSpec IFU, JWST/MIRI, VLT/KMOS, VLT/SINFONI and VLT/FLAMES-ARGUS.
* **Prepare the cube**: mask bad pixels/spikes, subtract the background (source-extractor based, or a
  supplied mask), PSF-match all wavelength channels, and extract a 1D spectrum from an aperture.
* **Fit a 1D spectrum** with the pre-built models for Halpha+[NII]+[SII], [OIII]+Hbeta (with optional
  FeII template), Halpha+[OIII] jointly, or the full optical line list - each with narrow-only,
  outflow, and single/broad-line-region (BLR) variants. You can also fit any function of your own
  via ``fitting_general``, or describe a custom set of lines declaratively with
  ``QubeSpec.Models.Custom_model.general_model`` (see :ref:`1D fitting <Fitting>`) instead of hand-writing one.
* **Fit spaxel-by-spaxel** using the same models (or your own), in parallel across CPUs, then
  post-process the results into flux/velocity/FWHM maps (see :ref:`spaxel_fitting`).
* **Inspect the results** interactively with the ``QubeSpec.Visualizations`` viewer.

Fitting uses ``emcee`` throughout, giving full posterior chains (and hence proper uncertainties) for
every fitted quantity, not just a point estimate.


Source and installation
-----------------------

QubeSpec is `developed at GitHub <https://github.com/honzascholtz/Qubespec>`_, and should be downloaded or pulled from there. Then, from inside your working directory, run (works on `bash`, untested on `tcsh`)

.. code:: bash

    conda create -n qubespec python=3.10
    conda activate qubespec
    pip3 install QubeSpec/.


You should then be able to import QubeSpec in python as:

.. code:: python

    import QubeSpec



Getting started
---------------

The best place to get started is by looking at the `iPython notebook examples <https://github.com/honzascholtz/Qubespec/tree/main/IFS_tutorial>`_. The full tutorial is in:


 - :ref:`Loading a cube  <Starting-with-QubeSpec>`: For Loading the IFS cube into QubeSpec and preparing it to fit 
 - :ref:`1D fitting <Fitting>`: Explanation of how fitting works. 
 - :ref:`Spaxel-by-Spaxel fitting <Spaxel_fitting>`: Fitting every spaxel in the cube. 


Acknowledgements
----------------

Loads of people.

.. toctree::
    :maxdepth: 1
    :hidden:

    QubeSpec_tutorial.rst
    Fitting.rst
    Spaxel_fitting.rst