This model code is designed to simulate in 1D the propagation of waves under ice causing deformation and fractures.

In order to be used, the config file (config.py) must first be edited to match the paths in use.
Libraries are either provided in the "dist" branch for oar or at: https://cloud.univ-grenoble-alpes.fr/s/xCSjrearNJNfr3C

The main scripts to be used are:
	- MultipleFloes1D.py: 	 Script to run an experiment using the initial monochromatic (omnipresent) waves as forcing
	- MultipleFloes1DSpec:  Script to run an experiment using spectral waves as forcing.
	  NOTE: 		 This also includes monochromatic wave fronts
	- MF1DSpecExp.py: 	 Script to run experiments using input and output files. This is used on oar.
	- SensStudy.py: 	 Script to run sensitivity studies for floe sizes
	- SensStudySaving.py: 	 Script to run sensitivity studies and save the data in pandas. This was designed and used by Alexandre Tlili during his internship to obtain the first floe size sensitivity study.

Ancillary python files with functions are:
	- FlexUtils_obj.py: 	 Functions that support the flexion, energy calculations and fracture of the ice
	- GenExp.py:		 Functions to create and modify experiments
	- IceDef.py:		 Definition of the "Floe" object and the necessary functions.
	- MF1D_func_saving.py:  Function to run experiments from SensStudySaving.py that save data in pandas. This was designed and used by Alexandre Tlili during his internship to obtain the first floe size sensitivity study.
	- MultipleFloes1D_func: Function to run experiments from SensStudy.py.
	- WaveDef.py:		 Definition of the "Wave" object and the necessary functions.
	- WaveSpecDef.py:	 Definition of the "Spectrum" object and the necessary functions.
	- treeForFrac.py:	 Functions to deal with the fracturation history.
	
Other files:
	- E_FlexDispTest.py:	Simple test to compare the vertical deformation versus flexion calculations of the elastic energy
	- pars.py:	File containing parameters to be used by the mode.
	  NOTE:	MF1DSpecExp is automated for oar and takes a lot of parameters from this file. 
	  		Other scripts (MultipleFloes[*] and SensStudy[*] are meant to be used by hand and so only take defining parameters (gravity, Young's modulus, Poisson's ratio, Fracture toughness)

How to use the model:
	There are two ways to use the model:
	- The production approach, on oar, using MF1DSpecExp to run many simulations and derive general results. Setup and run instructions are detailed in ReadMeOar.txt
	- The development approach, is to use the MultipleFloes1DSpec to define an experiment and run it locally.
	  Results, in the form of figures and text files, are found in subdirectories Figs and database, labeled using the parameters of the simulations. The databases are also used to avoid recomputation of simulations that were already completed.
	  Figures are generated within the scripts after the simulations are completed.
