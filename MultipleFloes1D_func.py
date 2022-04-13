#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""


def MF1D(**kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from FlexUtils_obj import PlotFloes, BreakFloes, PlotLengths, PlotFSD
    from WaveUtils import calc_k
    from WaveDef import Wave
    from IceDef import Floe

    growing = True
    reset = False

    # Wave Parameters
    n_0 = 0.2
    wvlength = 20
    beta = 0.1

    # Ice parameters
    h = 1
    x0 = 10
    L = 150
    DispType = 'Open'

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

    tw = np.linspace(0, wave.T, num=21)

    if reset:
        n_Loops = len(tw)
    else:
        n_Loops = 1

    FL = [0] * n_Loops

    for iL in range(n_Loops):
        t = np.arange(tw[iL], t_max + tw[iL], wave.T / 20)

        _ = wave.waves(x, t[0], floes=[floe1])  # assign waves over the whole domain

        floe1.calc_Eel(wave, t[0])
        Floes = [floe1]
        if not reset and growing:
            PlotFloes(x, t[0], Floes, wave)

        for it in range(len(t)):

            _ = wave.waves(x, t[it], floes=Floes)  # assign waves over the whole domain
            nF = len(Floes)
            Floes = BreakFloes(x, t[it], Floes, wave)
            if len(Floes) == nF and not reset and growing:
                PlotFloes(x, t[it], Floes, wave)
            if reset and growing and it % np.floor(len(t) / 10) == 0:
                print(f'{iL}-{it}')

        FL_temp = []
        for floe in Floes:
            FL_temp.append(floe.L)
        FL[iL] = FL_temp

    if reset:
        fig, hax = PlotLengths(tw, FL, wave, floe1.x0)
        if growing:
            lab = 'g'
        else:
            lab = '0'

        root = (f'FloeLengths_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
                f'h_{Floes[0].h:3.1f}_L0_{round(Floes[-1].xF[-1]-Floes[0].x0):02}')

        plt.savefig('FigsSum/' + root + '.png')

        fn = (f'_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
              f'h_{Floes[0].h:3.1f}_L0_{round(Floes[-1].xF[-1]-Floes[0].x0):02}')

        edges, values = PlotFSD(FL, wl=wvlength, h=h, n=n_0, DoSave=True, FileName=fn)

    return(edges, values)
