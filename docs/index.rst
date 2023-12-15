QubeSpec
========

QubeSpec is a simple but powerful python package to fit optical astronomical spectra and more importantly analysing IFS cube from JWST/NIRSpec, JWST/MIRI, VLT/KMOS and VLT/SINFONI. The code has built models for fitting Halpha, [OIII], Hbeta, [SII] and [NII] of galaxies, galaxies with outflows, Type-1 AGN and Quasars.

Authors: Jan Scholtz, Francesco D'Eugenio and Ignas Ignas Juodžbalis


What can QubeSpec do?
---------------------

#.. image:: images/sfh_from_spec.png

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

The best place to get started is by looking at the `iPython notebook examples <https://github.com/ACCarnall/bagpipes/tree/master/examples>`_. It's a good idea to tackle them in order as the later examples build on the earlier ones. These documentation pages contain a more complete reference guide.

Bagpipes is structured around three core classes:

 - :ref:`model_galaxy <making-model-galaxies>`: for generating model galaxy spectra
 - :ref:`galaxy <inputting-observational-data>`: for loading observational data into Bagpipes
 - :ref:`fit <fitting-observational-data>`: for fitting models to observational data.


Acknowledgements
----------------

Loads of people.