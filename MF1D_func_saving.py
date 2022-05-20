# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:32:58 2022

@author: tlilia
"""

import numpy as np

from FlexUtils_obj import BreakFloes
from WaveUtils import calc_k
from WaveDef import Wave
from IceDef import Floe


def MF1D_serie(param, **kwargs):
    '''
    Computes the resulting broken floes for parameters given in the pandas.Series param
    Inputs:
        param (pd.Series): parameters for simulation
        **kwargs:
            x0: initial position of the floe
            growing (bool): True if waves of growing amplitude
            phi0 (float): initial phase of the wave (rather use if growing = False)
            beta (float): growing rate of waves amplitude
    '''

    # Set fixed parameters
    x0 = 10
    growing = True
    phi0 = 0.
    beta = 0.1

    # Read inputs
    h = param['h']
    L = param['initialLength']
    n_0 = param['wvAmpl']
    wvLength = param['wvLength']

    DispType = param['DispType']
    EType = param['EnergyType']
    # nbFrac = param['Frac']

    # Initialize wave object
    if growing:
        wave = Wave(n_0, wvLength, beta=beta)
        t_max = 6 / beta
    else:
        wave = Wave(n_0, wvLength)
        t_max = 2 * wave.T

    # Initialize ice floe object
    floe1 = Floe(h, x0, L, DispType=DispType)
    floe1.kw = calc_k(1 / wave.T, h, DispType=DispType)

    # Initial setup and simulation time
    x = np.arange(2 * x0 + L)
    t = np.arange(0, t_max, wave.T / 20)

    # Initialize wave
    wave.phi = phi0
    _ = wave.waves(x, t[0], floes=[floe1])  # assign waves over the whole domain

    # Initialize floe
    floe1.calc_Eel(wave=wave, t=t[0], EType=EType)
    Floes = [floe1]

    # Simulation
    for it in range(len(t)):
        _ = wave.waves(x, t[it], floes=Floes)  # assign waves over the whole domain
        Floes = BreakFloes(x, t[it], Floes, wave, EType)

    # Computes the resulting floe lengths
    FloeSizes = np.empty(len(Floes))
    for k in range(len(Floes)):
        FloeSizes[k] = Floes[k].L

    return FloeSizes
