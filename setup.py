import subprocess
from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as file:
        return(file.read())

version = '1.0.0'

setup(name='QubeSpec',
      version=version,
      description='A library to optical/IFS spectra',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/honzascholtz/qubespec',
      author='Jan Scholtz',
      author_email='honzascholtz@gmail.com',
      license='Other/Proprietary License',
      packages=["QubeSpec", "QubeSpec.detection", "QubeSpec.Dust", "QubeSpec.Fitting", "QubeSpec.Maps", "QubeSpec.Models",
                "QubeSpec.Plotting","QubeSpec.Spaxel_fitting", "QubeSpec.Utils", "QubeSpec.Visualizations" ],
      install_requires=[
        'astropy>=5.2.0',
        'ipython>7.31.0',
        'matplotlib>=3.5.3',
        'numba>=0.56.3',
        'tqdm>=4.40.0',
        'emcee>=3.1.0',
        'brokenaxes>=0.5.0',
        'corner>=2.2.1',
        'scipy>=1.9.1',
        'multiprocess',
        'spectres',
        'photoutils',
        'sep'],
      python_requires='>=3.8.0',
      #entry_points={
      #  'console_scripts': [
      #      'inzimar = inzimar.inzimar:main',
      #      ],
      #},
      include_package_data=True,
      package_data={'': ['/jadify_temp/*', '/Models/FeII_templates/*']},
      zip_safe=False
     )
