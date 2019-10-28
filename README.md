
[![DOI](https://zenodo.org/badge/218070066.svg)](https://zenodo.org/badge/latestdoi/218070066)

# graphite

Typical workflow on my server as follows:

Load libraries:

$ source module

Setup parameters by editing 'input'

Run main project and create outputfolder 'test/':

$ mpirun -n 1 python main.py input test/

Run to get ARPES on the results. "L" for lesser Green's function (occupied states):

$ mpirun -n 1 python ARPES.py test/ L
