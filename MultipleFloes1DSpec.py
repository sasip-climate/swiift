#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:40 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
from FlexUtils_obj import PlotFloes, BreakFloes, PlotLengths, PlotFSD
from WaveUtils import calc_k, calc_cg
from WaveSpecDef import WaveSpec
from WaveChecks import plotDisp, plot_cg
from IceDef import Floe

DoPlots = True
SavePlots = True
repeats = 10

# Ice parameters
h = 1
x0 = 10
L = 200
DispType = 'ML'
EType = 'Flex'
# Initialize ice floe object
floe1 = Floe(h, x0, L, DispType=DispType)

# Wave Parameters
u = 5  # Wind speed (m/s)
# Initialize wave object
Spec = WaveSpec(u=u)

# calculate wave properties in ice
ki = calc_k(Spec.f, h, DispType=DispType)
ind = np.isnan(ki) + (abs(ki) > 10)
Spec.nf = sum(~ind)
Spec.f = Spec.f[~ind]
Spec.df = Spec.df[~ind]
Spec.Ei = Spec.Ei[~ind]
floe1.setWPars(Spec)

ki = ki[~ind]
cgi = calc_cg(ki, h, DispType=DispType)
xi = 1 / ((1 / 2) * h * ki ** 2)
if L > 5 * xi[6]:
    print('Warning: Floe is more than 5x the attenuation length of the peak')

plotDisp(Spec.f, h)
plot_cg(Spec.f, h)

# Initial setup
x = np.arange(2 * x0 + L + 1)
tProp = (2 * x0 / Spec.cgw[~ind] + L / cgi)
tPropMax = max(tProp)
tSpecM = max(tProp[Spec.Ei > 0.1 * max(Spec.Ei)])

Floes = [floe1]
# Visualize energy propagation in the domain
for t in np.arange(Spec.Tp, 2 * tSpecM + 1 / Spec.f[0], tSpecM / 10):
    Spec.calcExt(x, t, Floes)
    if t < tSpecM * 1.1:
        Spec.plotEx(fname=f'Spec/Spec_{DispType}_L0_{L:04}_{t:04.0f}.png', t=t)
    # Spec.set_phases(x, t, Floes)
    # Spec.plotWMean(x, floes=[floe1], fname='Spec/Waves_{t:04.0f}.png')

FL = [0] * repeats
t = np.arange(0, 2 * tSpecM + 1 / Spec.f[0], Spec.Tp / 20)

print(f'Launching {repeats} experiments with {round(t[-1])}s duration.')
for iL in range(repeats):

    Spec.calcExt(x, t[0], [floe1])
    # Spec.plotEx()
    Spec.set_phases(x, t[0], [floe1])

    wvf = Spec.calc_waves(floe1.xF)
    floe1.calc_Eel(wvf=wvf, EType=EType)
    Floes = [floe1]
    if repeats == 1:
        PlotFloes(x, t[0], Floes, Spec)

    for it in range(len(t)):

        nF = len(Floes)
        Spec.calcExt(x, t[it], Floes)
        # Spec.plotEx(t=t[it])
        Spec.set_phases(x, t[it], Floes)
        # Spec.plotWMean(x, floes=Floes)
        Floes = BreakFloes(x, t[it], Floes, Spec, EType)
        if DoPlots:
            for floe in Floes:
                floe.calc_Eel(wvf=Spec.calc_waves(floe.xF), EType=EType)
            if SavePlots:
                PlotFloes(x, t[it], Floes, Spec, f'Exp_{iL:02}_')
            else:
                PlotFloes(x, t[it], Floes, Spec)

    FL_temp = []
    for floe in Floes:
        FL_temp.append(floe.L)
    FL[iL] = FL_temp

wvlength = 2 * np.pi / calc_k(1 / Spec.Tp, floe1.h, DispType=floe1.DispType)
n0 = Spec.calcHs()

fig, hax = PlotLengths(np.arange(repeats), FL, x0=x0, h=h, xunits='trials')

root = (f'FloeLengths_Spec_{DispType}_n_{n0:1.2f}_l_{wvlength:06.2f}_'
        f'h_{h:3.1f}_L0_{L:04}_E_{EType}')

plt.savefig('FigsSum/' + root + '.png')

fn = (f'_Spec_{DispType}_n_{n0:05.2f}_l_{wvlength:06.2f}_'
      f'h_{h:3.1f}_L0_{L:04}_E_{EType}')

Lines = [[wvlength / 2, '$\lambda_p/2$']]
wvl_max = 2 * np.pi / calc_k(Spec.f[0], floe1.h, DispType=floe1.DispType)
Lines.append([wvl_max / 2, '$\lambda_{max}/2$'])
wvl_min = 2 * np.pi / calc_k(Spec.f[-1], floe1.h, DispType=floe1.DispType)
Lines.append([wvl_min / 2, '$\lambda_{min}/2$'])

PlotFSD(FL, h=h, n0=n0, FileName=fn, Lines=Lines)
