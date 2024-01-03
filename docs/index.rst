QubeSpec
========

QubeSpec is a simple but powerful python package to fit optical astronomical spectra and more importantly analysing IFS cube from JWST/NIRSpec, JWST/MIRI, VLT/KMOS and VLT/SINFONI. The code has built models for fitting Halpha, [OIII], Hbeta, [SII] and [NII] of galaxies, galaxies with outflows, Type-1 AGN and Quasars.

Authors: Jan Scholtz, Francesco D'Eugenio and Ignas Juodžbalis


What can QubeSpec do?
---------------------


Fit IFS cubes with different models. 


Source and installation
-----------------------

QubeSpec is `developed at GitHub <https://github.com/honzascholtz/Qubespec>`_, and should be downloaded or pulled from there. Then, from inside your working directory, run (works on `bash`, untested on `tcsh`)

.. code::
    conda create -n qubespec python=3.8 
    conda activate qubespec 
    pip3 install QubeSpec/.


You should then be able to import QubeSpec in python as:

.. code::
    import QubeSpec



Getting started
---------------

The best place to get started is by looking at the `iPython notebook examples <https://github.com/honzascholtz/Qubespec/tree/main/IFS_tutorial>`_. The full tutorial is in:


 - :ref:`Loading a cube  <QubeSpec_tutorial>`: For Loading the IFS cube into QubeSpec and preparing it to fit 
 - :ref:`1D fitting <Fitting>`: Explanation of how fitting works. 
 - :ref:`Spaxel-by-Spaxel fitting <Spaxel_fitting>`: Fitting every spaxel in the cube. 


Acknowledgements
----------------

Loads of people.

.. toctree::
    :maxdepth: 1
    :hidden:

    index.rst
    QubeSpec_tutorial.rst
    Fitting.rst
    Spaxel_fitting.rst