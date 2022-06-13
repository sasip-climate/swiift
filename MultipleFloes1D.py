#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

from FlexUtils_obj import PlotFloes, BreakFloes, BreakFloesStrain, PlotLengths, PlotFSD, PlotSum
from WaveUtils import calc_k
from WaveDef import Wave
from IceDef import Floe

multiFrac = 3
growing = True
reset = True
if growing:
    lab = 'g'
else:
    lab = '0'

# Wave Parameters
n_0 = 0.2
wvlength = 20
beta = 0.1

# Ice parameters
h = 1
x0 = 10
L = 150
DispType = 'Open'
EType = 'Flex'
FractureCriterion = 'Strain'  # 'Strain' or 'Energy'

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
    Evec = np.zeros([len(t), 1])

    wvf = wave.waves(x, t[0], floes=[floe1])  # over the whole domain

    floe1.calc_Eel(wave=wave, t=t[0], EType=EType)
    Floes = [floe1]
    if not reset and growing:
        PlotFloes(x, t[0], Floes, wave)

    tqdmlab = f'Time Loop {iL:02}' if n_Loops > 1 else 'Time Loop'
    for it in tqdm(range(len(t)), desc=tqdmlab):
        wvf = wave.waves(x, t[it], floes=Floes)  # over the whole domain
        nF = len(Floes)

        if FractureCriterion == 'Energy':
            Floes = BreakFloes(x, t[it], Floes, wave, multiFrac, EType)
        elif FractureCriterion == 'Strain':
            Floes = BreakFloesStrain(x, t[it], Floes, wave)
        else:
            raise ValueError('Non existing fracturation criterion')

        Evec[it] = (len(Floes) - 1) * Floes[0].k
        for floe in Floes:
            Evec[it] += floe.Eel
        if not reset:
            PlotFloes(x, t[it], Floes, wave)

    FL_temp = []
    for floe in Floes:
        FL_temp.append(floe.L)
    FL[iL] = FL_temp

if reset:
    fig, hax = PlotLengths(phi, FL, waves=wave, x0=x0, h=h)

    root = (f'FloeLengths_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
            f'h_{Floes[0].h:3.1f}_L0_{L:04}_'
            f'E_{EType}_B_{FractureCriterion}')

    plt.savefig(config.FigsDirSumry + root + '.png')

    fn = (f'_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
          f'h_{Floes[0].h:3.1f}_L0_{L:04}_'
          f'E_{EType}_B_{FractureCriterion}')

    PlotFSD(FL, wl=wvlength, h=h, n0=n_0, DoSave=True, FileName=fn)
else:
    PlotSum(t, Evec, leg=[EType])
    root = (f'Energy_Time_Series_{lab}_{DispType}_n_{wave.n0:3}_l_{wave.wl:2}_'
            f'h_{Floes[0].h:3.1f}_L0_{L:04}_'
            f'E_{EType}_B_{FractureCriterion}')

    plt.savefig(config.FigsDirSumry + root + '.png')
