# SWIIFT
Surface Wave Impact on sea Ice---Fracture Toolkit

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

A numerical model to study wave-induced fracture of sea ice floes.

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Acknowledgements](#acknowledgements)

## Installation

Make sure your Python version is at least 3.11.
In a terminal, run 
```console
python -m venv .venv
source ./venv/bin/activate
python -m pip install --upgrade pip
python -m pip install git+ssh://git@github.com/sasip-climate/swiift.git#egg=swiift
```
to install the latest version of `SWIIFT` and its dependencies in a
dedicated environment.

Alternatively, a specific version can be installed by specifying it in the git URL,
```console
python -m pip install git+ssh://git@github.com/sasip-climate/swiift.git@$ver#egg=swiift
```
where `$var` is to be replaced by the desired version tag (for instance, `v0.4.0`).

> [!NOTE]
> Depending on your system, you might have to replace the command `python` with
> `python3`, or even a specific minor version such as `python3.11`.

## Usage

### Setting up an experiment

### Running an experiment

### Helper classes


## Development

### Using Hatch

This project is managed with [Hatch](https://github.com/pypa/hatch).
Refer to [its documentation](https://hatch.pypa.io/latest/install) for installation.

Once Hatch is installed, run
```console
git clone git@github.com:sasip-climate/swiift.git
cd swiift
hatch test --all --parallel
```
to install `SWIIFT` along with its dependencies, and run the test suite in
a dedicated environment.


## Acknowledgements

This work is part of the [SASIP project](https://sasip-climate.github.io/),
funded by Schmidt Sciences.
