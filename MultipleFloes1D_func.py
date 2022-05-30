#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from FlexUtils_obj import PlotFloes, BreakFloes, PlotLengths, PlotFSD
from WaveUtils import calc_k
from WaveDef import Wave
from IceDef import Floe
from tqdm import tqdm


def MF1D(**kwargs):
    # Variable to control plots to be made
    # 0: None, 1: Lengths, 2: Lengths and FSD, 3: Lengths, FSD and Floes
    DoPlots = 1
    FigsDirSumry = config.FigsDirSumry

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

        _ = wave.waves(x, t[0], floes=[floe1])  # assign waves over the whole domain

        floe1.calc_Eel(wave=wave, t=t[0], EType=EType)
        Floes = [floe1]
        if not reset and growing:
            PlotFloes(x, t[0], Floes, wave)

        for it in tqdm(range(len(t))):

            _ = wave.waves(x, t[it], floes=Floes)  # assign waves over the whole domain
            Floes = BreakFloes(x, t[it], Floes, wave, EType)
            if not reset and DoPlots > 2:
                PlotFloes(x, t[it], Floes, wave)

        FL_temp = []
        for floe in Floes:
            FL_temp.append(floe.L)
        FL[iL] = FL_temp

    if reset:
        if DoPlots > 0:
            fig, hax = PlotLengths(phi, FL, waves=wave, x0=x0, h=h)
            if growing:
                lab = 'g'
            else:
                lab = '0'

            root = (f'FloeLengths_{lab}_{DispType}_n_{wave.n0:3}_wl_{wave.wl:02}_'
                    f'h_{Floes[0].h:3.1f}_L0_{L:04}_'
                    f'E_{EType}')

            plt.savefig(FigsDirSumry + root + '.png')

        if DoPlots > 1:
            fn = (f'_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
                  f'h_{Floes[0].h:3.1f}_L0_{L:04}_'
                  f'E_{EType}')

            edges, values = PlotFSD(FL, wl=wvlength, h=h, n=n_0, DoSave=True, FileName=fn)
        else:
            Ll = []
            for l in FL:
                Ll += l
            values, edges = np.histogram(Ll, bins=np.arange(1, round(max(Ll)) + 1))

        return(FL, edges, values)
    else:
        return(FL)
