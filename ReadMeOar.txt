1 - Connecting to OAR:
	OAR can be accessed via ssh using the same username as for UGA emails, connecting to ist-oar.ujf-grenoble.fr.
	It is necessary to be wired connected properly to the ISTerre network or use the vpn (https://nomadisme.grenet.fr/installVPN.htm) for a secure connection in order to reach OAR by ssh.

2 - Connecting OAR to the UGA gitlab
	In order to interact with the gitlab, an ssh key is necessary. It can be the same as for your regular workstation if you copy your private key from your local computer to OAR (presumably in ~/.ssh/)
	OAR does not have a running ssh agent by default, so it needs to be activated and given the key before interacting with the gitlab:
	eval `ssh-agent -s`
	ssh-add ~/.ssh/key_uga_gitlab

3 - Getting the MF1D code for OAR
	The MF1D code in the oar branch is intended for use on oar, with some tweaks that allow reference to code in different directories.
	The ideal way to maintain the oar branch is through rebase-pulling master into it, and keeping the few changes made specifically for OAR in the last commits of the log.

4 - Using the MF1D code on OAR
	OAR is meant for large numerical experiments. As such, launching individual jobs is not recommended.
	To execute python code on OAR, loading the python3 module is recommended using: module load python/python3.9
	This can be done to quickly launch a python script on the head note to test (Note: don't leave it running for long as the head node is NOT intended to run code. This should be used only to see if things initialize properly)
	Running a python script in the terminal is done simply with: python3 Script.py
	The general way to use OAR is by submitting jobs. In order to do this, two bash shell scripts are provided:
		- MF1Dtest.sh
		- MF1Dsub.sh
	As implied by their names, they are intended to test code, and submit simulations. To do this:
		- oarsub -S -n NameOfTest --project iste-equ-failles ./MF1Dtest.sh -p "dedicated='none'" -t devel
		- oarsub -S -n NameOfTest --project iste-equ-failles ./MF1Dsub.sh
		(Note: the test queue only allows for jobs of up to one hour and generally has quicker access to resources. There isn't otherwise much difference)

5 - Running experiments on OAR
	Experiments should be setup prior, by creating a series of folders with their own pars.py inside.
	Note: If a pars.py exists in the code or libraries folder, it may be used instead of the local one, meaning all simulations will be exactly the same.
	Four scripts are provided to manage experiments:
		Two run management scripts, which setup the necessary shortcuts and submit jobs
		Both scripts require the name of the directory containing all the experiment directories and, optionally, a number of accepted simultaneous jobs
		- subTest.sh: Submit a series of test runs, running on two cores for one hour
		- subExp.sh: Submit experiments, running on twelve cores for 4 hours
		Three Ancillary scripts:
		- arcExp.sh: Look into all of the experiment directories of a given directory, archiving their content and moving it to the archive drive on oar
		- checkExp.sh: Look into all of the experiment directories of a given directory, reporting their status (number of jobs completed, requested and, if incomplete, any error message)
		- cleanExp.sh: remove all the outputs, linked files and necessary subdirectories in each experiment folder. Leaving them essentially in their initial state with only pars.py

6 - Example Folder tree
	Home
	|-> PythonCode
	      |-> All the content of the git repo
	      |-> Librairies
	|-> Experiments
	      |-> Bash scripts to submit jobs on OAR
	      |-> Sens1
	            |-> Bazillion folders for the sensitivity study
	                 |-> pars.py
	                 |-> Automatically generated links to scripts, code and output folder
	      |-> Test
	            |-> A few folders to run tests
	                 |-> pars.py
	                 |-> Automatically generated links to scripts, code and output folder

6 - Example code to setup and run an experiment on oar from a working installation of the model locally:
	Note: Don't copy and paste as it will not work, but use for inspiration
	Note2: Use two terminals, one to ssh on oar (unmarked lines), one to run commands locally (identified)
	
	On your local computer, design an experiment. This is done in python using functions in GenExp.py The setup for the introductory paper on this model, SetupPaper1.py, can be used as an example.
	(from your local computer, this terminal will now be the oar terminal:) ssh ist-oar.ujf-grenoble.fr
	mkdir .ssh
	(from your local computer:) scp ~/.ssh/key_uga_gitlab ist-oar.ujf-grenoble.fr:.ssh/
	eval `ssh-agent -s`
	ssh-add ~/.ssh/key_uga_gitlab
	mkdir PythonCode
	git clone git@gricad-gitlab.univ-grenoble-alpes.fr:auclaije/flex-frac-1d.git PythonCode
	cd PythonCode
	mkdir Libraries
	(From your local computer:) scp path/to/code/librairies oar:PythonCode/Librairies/
	git checkout dist
	vi config.py # Change the paths at the beginning to refer to ones
	cd ../
	mkdir Experiments
	(From your local computer:) scp -r path/to/Experiments/folder/ExpFolderName Experiments/ExpFolderName
	(From your local computer:) scp /path/to/oar/scripts/folder/*.sh Experiments
	cd Experiments
	nohup ./subExp.sh ExpFolderName 12 > Sens1.out &
	(wait a bit)
	oarstat -u $USER # To check if something got submitted
	tail -f Sens1.out # To see where the script is currently
	(From your local computer:) scp -r /path/to/oar/archives/ExpFolderName Experiments/ExpFolderName # Copy archives to local computer
	(From your local computer:) ./extractExp.sh ExpFolderName
	(From your local computer:) run analysis scripts (examples of which are provided for the introductory paper on this model in the Paper1 folder, by section, named Paper1_[Section]_Reanalysis.py
