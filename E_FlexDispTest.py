#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""

import numpy as np
from FlexUtils_obj import PlotFloes, BreakFloes, PlotSum
from WaveUtils import calc_k, SpecVars
from WaveDef import Wave
from IceDef import Floe

growing = True
reset = False
stop = True

# Wave Parameters
(n_0, _, _, _, wvlength) = SpecVars(5)  # Spectral parameters from wind speeds in m/s
n_0 = 2
wvlength = 10
beta = 0.1

# Ice parameters
h = 1
x0 = 10
L = 10
DispType = 'Open'

# Initialize wave object
if growing:
    wave = Wave(n_0, wvlength, beta=beta)
    t_max = 6 / beta
else:
    wave = Wave(n_0, wvlength)
    t_max = 2 * wave.T

# Initialize ice floe object
floe1 = Floe(h, x0, L, DispType=DispType)
floe1.kw = calc_k(1 / wave.T, h, DispType=DispType)

# Initial setup
x = np.arange(2 * x0 + L)

phi = 2 * np.pi * np.linspace(0, 1, num=21)

if reset:
    n_Loops = len(phi)
else:
    n_Loops = 1

FL = [0] * n_Loops

t = np.arange(0, t_max, wave.T / 20)
for iL in range(n_Loops):
    wave.phi = phi[iL]
    Evec = np.zeros([len(t), 2])

    Floes = [floe1]
    nF = 1

    for it in range(len(t)):

        wvf = wave.waves(x, t[it], floes=Floes)  # over the whole domain

        if len(Floes) == nF:
            for floe in Floes:
                floe.calc_Eel(wave=wave, t=t[it], EType='Disp')
            PlotFloes(x, t[it], Floes, wave)

        if nF == 1:
            # floe1.calc_w(wave.waves(floe1.xF, t[0], amp=floe1.a0, phi=floe1.phi0, floes=[floe1]))
            # floe1.calc_du(f'FitFigs/Fits_x0_{floe1.x0}_L_{floe1.L:02.2f}_t_0')
            Evec[it, 1] = floe1.calc_Eel(wave=wave, t=t[0], EType='Disp')
            Evec[it, 0] = floe1.calc_Eel(wave=wave, t=t[0], EType='Flex')
            itf = it
            print(f'Elastic energy at step {it:02} with amplitude {wave.amp(t[it]):3.2f}: '
                  f'Flexion: {Evec[it,0]:04.3f} - Displacement: {Evec[it,1]:04.3f}')
        else:
            if stop:
                break

        nF = len(Floes)
        Floes = BreakFloes(x, t[it], Floes, wave, 'Flex')

        # for floe in Floes:
        #     floe.calc_du(f'FitFigs/Fits_x0_{floe.x0}_L_{floe.L:02.2f}_t_{it}')

    PlotSum(t[:itf], Evec[:itf], leg=['Flexion', 'Displacement'])
