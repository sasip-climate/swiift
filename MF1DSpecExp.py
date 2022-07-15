#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from os import path
# from tqdm import tqdm
import time
from prog_rep import prog_rep
from FlexUtils_obj import PlotFloes, PlotLengths, PlotFSD
from FlexUtils_obj import BreakFloes, BreakFloesStrain, getFractureHistory
from WaveUtils import calc_k
from WaveSpecDef import WaveSpec
from WaveChecks import plotDisp, plot_cg
from IceDef import Floe
import pars

# 0: None, 1: Lengths, 2: Lengths and FSD, 3: Lengths, FSD and saved Floes, 4: Lengths, FSD and Floes
DoPlots = 3
repeats = 20
multiFrac = pars.multiFrac
FractureCriterion = pars.FractureCriterion

# Ice parameters
h = pars.h
x0 = 50
L = 100
dx = 0.5
DispType = 'ML'
EType = 'Flex'
# Initialize ice floe object
floe1 = Floe(h, x0, L, DispType=DispType, dx=dx)

# Wave Parameters
u = pars.u  # Wind speed (m/s)
# Initialize wave object
Spec = WaveSpec(u=u)

# Single frequency wave front
# n0 = pars.n0
# Hs = (2 * n0) ** 0.5
# Spec = WaveSpec(f=pars.f, Hs=Hs)

# calculate wave properties in ice
Spec.checkSpec(floe1)
ki = floe1.kw
floe1.setWPars(Spec)

xi = 1 / floe1.alpha
if L > 5 * xi[Spec.f == Spec.fp]:
    print('Warning: Floe is more than 5x the attenuation length of the peak', flush=True)

plotDisp(Spec.f, h)
plot_cg(Spec.f, h)

# Initial setup
x = np.arange(x0 + L + 2)
xProp = 4 / floe1.alpha
xProp[xProp > L] = L
tProp = (2 * x0 / Spec.cgw + xProp / floe1.cg)
tPropMax = max(tProp)
tSpecM = max(tProp[Spec.Ei > 0.1 * max(Spec.Ei)])

# Necessery, as defining filename roots calls Floes which may not exist if all experiments already saved
Floes = [floe1]

# Visualize energy propagation in the domain
if DoPlots > 0:
    for t in np.arange(Spec.Tp, 2 * tSpecM + 1 / Spec.f[0], tSpecM / 10):
        Spec.calcExt(x, t, Floes)
        if t < tSpecM * 1.1:
            Spec.plotEx(fname=(config.FigsDirSpec + f'Spec_{DispType}_L0_{L:04}_{t:04.0f}.png'), t=t)
        # Spec.set_phases(x, t, Floes)
        # Spec.plotWMean(x, floes=[floe1], fname='Spec/Waves_{t:04.0f}.png')

FL = [0] * repeats

dt = dx / (Spec.fp * Spec.wlp)
t = np.arange(0, 1.2 * tSpecM + 2 / Spec.f[0], Spec.Tp / 20)  # min(Spec.Tp / 20, dt))

phi = 2 * np.pi * np.linspace(0, 1, num=repeats, endpoint=False)

print(f'Launching {repeats} experiments:')
for iL in range(repeats):
    LoopName = f'Exp_{iL:02}_E_{EType}_F_{FractureCriterion}_h_{h:3.1f}m_Hs_{Spec.Hs:04.1f}m.txt'
    DataPath = config.DataTempDir + LoopName
    FracHistPath = DataPath[:-4] + '_History.txt'
    if path.isfile(DataPath):
        print(f'Reading existing data for loop {iL:02}', flush=True)
        FL[iL] = list(np.loadtxt(DataPath, ndmin=1))
        history = None
        continue

    # Change the phases of each wave
    if len(Spec.f) == 1:
        Spec.phi = np.array([phi[iL]])
    Spec.setWaves()

    Spec.calcExt(x, t[0], [floe1])
    # Spec.plotEx()
    Spec.set_phases(x, t[0], [floe1])

    Floes = [floe1]

    tqdmlab = f'Time Loop {iL:02}' if repeats > 1 else 'Time Loop'
    start_time = time.time()
    print(tqdmlab + ': ', end='', flush=True)
    for it in range(len(t)):  # tqdm(range(len(t)), desc=tqdmlab):

        nF = len(Floes)

        Spec.calcExt(x, t[it], Floes)
        # Spec.plotEx(t=t[it])
        Spec.set_phases(x, t[it], Floes)
        # Spec.plotWMean(x, floes=Floes)
        try:
            if FractureCriterion == 'Energy':
                Floes = BreakFloes(x, t[it], Floes, Spec, multiFrac, EType)
            elif FractureCriterion == 'Strain':
                Floes = BreakFloesStrain(x, t[it], Floes, Spec)
            else:
                raise ValueError('Non existing fracturation criterion')
        except np.linalg.LinAlgError:
            print('Time Loop aborted due to singular matrix', flush=True)
            break

        if DoPlots > 3:
            PlotFloes(x, t[it], Floes, Spec)
        elif DoPlots > 2 or len(Floes) > nF:
            lab = f'Exp_{iL:02}_E_{EType}_F_{FractureCriterion}'
            PlotFloes(x, t[it], Floes, Spec, lab, it)

        if Floes[-1].x0 > 0.6 * L + x0:
            # Update floes in history (NOTE: It also update the actual floe's length)
            getFractureHistory().modifyLengthDomain(L / 2)
            # Update floe resolution and matrix
            nx = max(int(np.ceil(Floes[-1].L)), 100)
            Floes[-1].xF = Floes[-1].x0 + np.linspace(0, Floes[-1].L, nx)
            Floes[-1].initMatrix()
            # And update domain
            x = np.arange(0, x[-1] + L / 2 + 1, 1)
            L *= 1.5
            print('+', end='')

        prog_rep(it, len(t) - 1, flush=True)

    FL_temp = []
    for floe in Floes:
        FL_temp.append(floe.L)
    np.savetxt(DataPath, np.array(FL_temp))
    FL[iL] = FL_temp

    if DoPlots > 2:
        DoPlots = 2

    history = getFractureHistory()
    if DoPlots > 0 and (not history is None):
        fGen = config.FigsDirSumry + 'Gen' + LoopName[3:-4] + '.png'
        history.plotGeneration(filename=fGen)
        if DoPlots > 3:
            history.plotGeneration()
        # Save the fracture history, giving (L, x0, gen, time, boolean existing, parent) informations for each floe that has existed
        np.savetxt(FracHistPath, history.asArray())

    print(f'Time taken: {round(time.time() - start_time)}s', flush=True)

n0 = Spec.calcHs()

if DoPlots > 0:
    if len(Spec.f) == 1:
        xv = phi
        xu = 'phase'
    else:
        xv = np.arange(repeats)
        xu = 'trials'

    root = (f'FloeLengths_Spec_E_{EType}_F_{FractureCriterion}_'
            f'{DispType}_Hs_{Spec.Hs:05.2f}_wlp_{Spec.wlp:06.2f}_h_{h:3.1f}_L0_{L:04}')

    fig, hax = PlotLengths(xv, FL, x0=x0, h=h, Spec=Spec, xunits=xu)
    plt.savefig(config.FigsDirSumry + root + '.png', dpi=150)

    fig, hax = PlotLengths(xv, FL, x0=x0, h=h, Spec=Spec, xunits=xu, trim=True)
    plt.savefig(config.FigsDirSumry + root + '_trim.png', dpi=150)

if DoPlots > 1:
    fn = (f'_Spec_E_{EType}_F_{FractureCriterion}_'
          f'{DispType}_Hs_{Spec.Hs:05.2f}_wlp_{Spec.wlp:06.2f}_h_{h:3.1f}_L0_{L:04}')

    wvl_lab = '$\lambda_p/2$' if len(Spec.f) > 1 else '$\lambda/2$'
    Lines = [[Spec.wlp / 2, wvl_lab]]
    wvl_max = 2 * np.pi / calc_k(Spec.f[0], floe1.h, DispType=floe1.DispType)
    if wvl_max > Spec.wlp * 1.5:
        Lines.append([wvl_max / 2, '$\lambda_{max}/2$'])
    wvl_min = 2 * np.pi / calc_k(Spec.f[-1], floe1.h, DispType=floe1.DispType)
    if wvl_min < Spec.wlp / 1.5:
        Lines.append([wvl_min / 2, '$\lambda_{min}/2$'])

    PlotFSD(FL, h=h, n0=n0, FileName=fn, Lines=Lines, Spec=Spec)
