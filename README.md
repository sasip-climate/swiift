# flexfrac1d

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A numerical model to study wave-induced fracture of sea ice floes.

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Acknowledgements](#acknowledgements)

## Installation

Make sure your Python version is at least 3.10,
and run (replacing `$folder` with your desired directory name)
```console
mkdir $folder
cd $folder
python -m venv .venv
source ./venv/bin/activate
python -m pip install --upgrade pip
python -m pip install git+ssh://git@github.com/sasip-climate/FlexFrac1D.git#egg=flexfrac1d
```
to install FlexFrac1D and its dependencies in a dedicated environment.


## Usage

### Setting up an experiment

### Running an experiment

### Helper classes


## Development

This project is managed with [Hatch](https://github.com/pypa/hatch).
Refer to [its documentation](https://hatch.pypa.io/latest/install) for installation.

Once Hatch is installed, run
```console
git clone git@github.com:sasip-climate/FlexFrac1D.git
cd FlexFrac1D
hatch env create
hatch shell
python -m pip install -r requirements.txt
```
to install FlexFrac1D and its dependencies in a development environment.

You can then run
```console
python -m pytest
```
to run the test suite.


## Acknowledgements

This work is part of the [SASIP project](https://sasip-climate.github.io/),
funded by Schmidt Sciences.
