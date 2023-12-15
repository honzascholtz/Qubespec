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
      packages=find_packages(),
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
      package_data={'': ['MSA/MSA.py', 'MSA/_init_.py','detection/Detection.py','Model/Custom_model.py','Models/FeII_comp.py','Models/Halpha_models.py','Models/Halpha_OIII_models.py','Models/OIII_models.py','Models/QSO_models.py',
          'Models/FeII_templates/FeII_Tsuzuki_opttemp.txt', 'Models/FeII_templates/bg92.con',
          'Models/FeII_templates/Veron-cetty_2004.fits', 'jadify_temp/red_table_template.csv',
          'jadify_temp/r100_jades_deep_hst_v3.1.1_template.yaml', 'jadify_temp/Temp_prism_clear_v3.0_extr3_1D.fits']},
      zip_safe=False
     )
