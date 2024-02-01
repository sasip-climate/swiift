#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:55:33 2022

@author: auclaije
"""

import numpy as np

# Note: water and ice parameters from a review paper by Squire et al (1995)
# General parameters
g = 9.8
rho_w = 1025
rho_i = 922.5

E = 6e9  # Elastic modulus
v = 0.3  # Poisson's ratio
K = 1e5  # Fracture energy

# Critical strain supported by ice before fracture -> Dumont2011/Roach2018
strainCrit = 3e-5

h = 1  # ice thickness (m)
u = 5  # wind speed (m/s)
f = 0.25  # frequency (Hz)
wl = g / (2 * np.pi * f**2)
SpecType = "JONSWAP"
tail_fac = 2  # wavelength factor for exponential decay past the last energetic point
n = -2  # power law exponent
n0 = 1  # wave amplitude (m)

FractureCriterion = "Energy"
multiFrac = True

N = 101
Deriv101 = (
    6 * np.eye(N)
    - 4 * np.eye(N, k=1)
    - 4 * np.eye(N, k=-1)
    + np.eye(N, k=2)
    + np.eye(N, k=-2)
)
# First two rows can't use centered difference
Deriv101[0, [0, 1, 2]] = np.array([2, -4, 2])
Deriv101[1, [0, 1, 2, 3]] = np.array([-2, 5, -4, 1])
# Last two rows can't use centered difference either
Deriv101[-1, [-3, -2, -1]] = np.array([2, -4, 2])
Deriv101[-2, [-4, -3, -2, -1]] = np.array([1, -4, 5, -2])
