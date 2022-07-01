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
from os import path

from FlexUtils_obj import PlotFloes, PlotLengths, PlotFSD, PlotSum
from FlexUtils_obj import BreakFloes, BreakFloesStrain, getFractureHistory
from WaveUtils import calc_k, omega
from WaveDef import Wave
from IceDef import Floe


# Variable to control plots to be made
# 0: None, 1: Lengths, 2: Lengths and FSD, 3: Lengths, FSD and saved Floes, 4: Lengths, FSD and Floes
DoPlots = 3
multiFrac = 2
growing = True
reset = True
if growing:
    lab = 'g'
else:
    lab = '0'

# Wave Parameters
n_0 = 0.2
wvlength = 20

# Ice parameters
h = 1
x0 = 10
L = 150
dx = 0.5
DispType = 'Open'
EType = 'Flex'
FractureCriterion = 'Energy'  # 'Strain' or 'Energy'

# Initialize wave object
if growing:
    beta = omega((2 * np.pi / wvlength)) * wvlength / (2 * np.pi * L)
    wave = Wave(n_0, wvlength, beta=beta)
    t_max = 6 / beta
else:
    wave = Wave(n_0, wvlength)
    t_max = 2 * wave.T

# Initialize ice floe object
floe1 = Floe(h, x0, L, DispType=DispType, dx=dx)
floe1.kw = calc_k(1 / wave.T, h, DispType=DispType)

# Initial setup
x = np.arange(2 * x0 + L)

# Necessery, as defining filename roots calls Floes which may not exist if all experiments already saved
Floes = [floe1]

if reset:
    phi = 2 * np.pi * np.linspace(0, 1, num=21)
    n_Loops = len(phi)
else:
    phi = [0]
    n_Loops = 1

FL = [0] * n_Loops

dt = dx / (5 * wave.omega / wave.k)

t = np.arange(0, t_max, dt)  # wave.T / 20)

print(f'Launching {n_Loops} experiments:')
for iL in range(n_Loops):
    LoopName = f'Exp_{iL:02}_E_{EType}_F_{FractureCriterion}_h_{h:3.1f}m_n0_{wave.n0:04.1f}m.txt'
    DataPath = config.DataTempDir + LoopName
    if path.isfile(DataPath):
        print(f'Reading existing data for loop {iL:02}')
        FL[iL] = list(np.loadtxt(DataPath))
        history = []
        Evec = np.zeros([len(t), 1])
        continue

    wave.phi = phi[iL]
    Evec = np.zeros([len(t), 1])
    Floes = [floe1]

    _ = wave.waves(x, t[0], floes=Floes)  # assign waves over the whole domain

    if not reset and growing:
        PlotFloes(x, t[0], Floes, wave)

    tqdmlab = f'Time Loop {iL:02}' if n_Loops > 1 else 'Time Loop'
    for it in tqdm(range(len(t)), desc=tqdmlab):

        _ = wave.waves(x, t[it], floes=Floes)  # over the whole domain
        nF = len(Floes)

        if FractureCriterion == 'Energy':
            Floes = BreakFloes(x, t[it], Floes, wave, multiFrac, EType)
            Evec[it] = (len(Floes) - 1) * Floes[0].k
            for floe in Floes:
                Evec[it] += floe.Eel
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
    np.savetxt(DataPath, np.array(FL_temp))
    FL[iL] = FL_temp

if DoPlots > 0 and len(FL[0]) > 1:
    fig, hax = PlotLengths(phi, FL, waves=wave, x0=x0, h=h)

    root = (f'FloeLengths_E_{EType}_F_{FractureCriterion}_{lab}_'
            f'{DispType}_n_{wave.n0:3}_wl_{wave.wl:02.1f}_h_{Floes[0].h:03.1f}_L0_{L:04}')

    plt.savefig(config.FigsDirSumry + root + '.png', dpi=150)

if DoPlots > 1:
    if reset:
        fn = (f'_E_{EType}_F_{FractureCriterion}_{lab}_'
              f'{DispType}_n_{wave.n0:3}_wl_{wave.wl:2}_h_{Floes[0].h:3.1f}_L0_{L:04}')

        PlotFSD(FL, wl=wvlength, h=h, n0=n_0, FileName=fn)
    else:
        PlotSum(t, Evec, leg=[EType])
        root = (f'Energy_Time_Series__E_{EType}_F_{FractureCriterion}_{lab}_'
                f'{DispType}_n_{wave.n0:3}_wl_{wave.wl:2}_h_{Floes[0].h:3.1f}_L0_{L:04}')
        history = getFractureHistory()
        if history != [] and history.floe != []:
            history.plotGeneration()

        plt.savefig(config.FigsDirSumry + root + '.png', dpi=150)
