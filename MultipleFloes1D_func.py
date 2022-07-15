#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from FlexUtils_obj import PlotFloes, BreakFloes, BreakFloesStrain, PlotLengths, PlotFSD
from WaveUtils import calc_k
from WaveDef import Wave
from IceDef import Floe


def MF1D(**kwargs):
    # Variable to control plots to be made
    # 0: None, 1: Lengths, 2: Lengths and FSD, 3: Lengths, FSD and saved Floes, 4: Lengths, FSD and Floes
    DoPlots = 1
    FigsDirSumry = config.FigsDirSumry
    multiFrac = False
    FractureCriterion = 'Energy'

    growing = True
    reset = True

    # Wave Parameters
    n_0 = 0.2
    wvlength = 20
    beta = 0.1

    # Ice parameters
    h = 1
    x0 = 10
    L = 150
    DispType = 'Open'
    EType = 'Disp'

    for key, value in kwargs.items():
        if key == 'growing':
            growing = value
        elif key == 'reset':
            reset = value
        elif key == 'n_0':
            n_0 = value
        elif key == 'wvlength':
            wvlength = value
        elif key == 'beta':
            beta = value
        elif key == 'h':
            h = value
        elif key == 'x0':
            x0 = value
        elif key == 'L':
            L = value
        elif key == 'DispType':
            DispType = value
        elif key == 'EType':
            EType = value
        elif key == 'DoPlots':
            DoPlots = value
        elif key == 'SaveDirectory':
            FigsDirSumry = value
        elif key == 'multiFrac':
            multiFrac = value
        elif key == 'FracCrit':
            FractureCriterion = value

    # Initialize wave object
    if growing:
        wave = Wave(n_0, wvlength, beta=beta)
        t_max = 6 / beta
        lab = 'g'
    else:
        wave = Wave(n_0, wvlength)
        t_max = 2 * wave.T
        lab = '0'

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
        Floes = [floe1]

        _ = wave.waves(x, t[0], floes=Floes)  # assign waves over the whole domain

        if not reset and growing:
            PlotFloes(x, t[0], Floes, wave)

        for it in range(len(t)):

            _ = wave.waves(x, t[it], floes=Floes)  # assign waves over the whole domain
            nF = len(Floes)

            if FractureCriterion == 'Energy':
                Floes = BreakFloes(x, t[it], Floes, wave, multiFrac, EType)
            elif FractureCriterion == 'Strain':
                Floes = BreakFloesStrain(x, t[it], Floes, wave)
            else:
                raise ValueError('Non existing fracturation criterion')

            if DoPlots > 3:
                PlotFloes(x, t[it], Floes, wave)
            elif DoPlots > 2 or len(Floes) > nF:
                Explab = f'Exp_{iL:02}_E_{EType}_F_{FractureCriterion}_{lab}'
                PlotFloes(x, t[it], Floes, wave, Explab, it)

        FL_temp = []
        for floe in Floes:
            FL_temp.append(floe.L)
        FL[iL] = FL_temp

    if reset:
        if DoPlots > 0:
            fig, hax = PlotLengths(phi, FL, waves=wave, x0=x0, h=h)

            root = (f'FloeLengths_E_{EType}_F_{FractureCriterion}_{lab}_'
                    f'{DispType}_n_{wave.n0:3}_wl_{wave.wl:02.1f}_h_{h:03.1f}_L0_{L:04}')

            plt.savefig(FigsDirSumry + root + '.png', dpi=150)

        if DoPlots > 1:
            fn = (f'_E_{EType}_F_{FractureCriterion}_{lab}_'
                  f'{DispType}_n_{wave.n0:3}_wl_{wave.wl:02.1f}_h_{h:03.1f}_L0_{L:04}')

            edges, values = PlotFSD(FL, wl=wvlength, h=h, n=n_0, FileName=fn)
        else:
            Ll = []
            for l in FL:
                Ll += l
            values, edges = np.histogram(Ll, bins=np.arange(1, round(max(Ll)) + 1))

        return(FL, edges, values)
    else:
        return(FL)
