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

growing = False
reset = True

# Wave Parameters
u = 10  # Wind speed (m/s)
# Initialize wave object
Spec = WaveSpec(u=u)

# Ice parameters
h = 1
x0 = 450
L = 1500
DispType = 'ElML'
# Initialize ice floe object
floe1 = Floe(h, x0, L, DispType=DispType)

# calculate wave properties in ice
Spec.ki = calc_k(Spec.f, h, DispType=DispType)
Spec.cgi = calc_cg(Spec.ki, h, DispType=DispType)
plotDisp(Spec.f, h)
plot_cg(Spec.f, h)

# Initial setup
x = np.arange(2 * x0 + L + 1)
tProp = max(2 * x0 / Spec.cgw + L / Spec.cgi)

# Visualize energy propagation in the domain
for t in np.arange(10, 2 * tProp + 1 / Spec.f[0], 10):
    Spec.calcExt(x, t, [floe1])
    if t < tProp * 1.1:
        Spec.plotExt(x)
        plt.savefig(f'Spec/Spec_{t:04.0f}.png')
    Spec.plotWMean(x, t, floes=[floe1], DoSave=True)

exit

FL = [0] * repeats

for iL in range(repeats):
    t = np.arange(0, 2 * tProp + 1 / Spec.f[0], Spec.Tp / 20)

    Spec.calcExt(x, t, [floe1])
    wvf = Spec.waves(floe1.xf, t, floes=[floe1])

    floe1.calc_Eel(wvf)
    Floes = [floe1]
    if not reset and growing:
        PlotFloes(x, t[0], Floes, Spec)

    for it in range(len(t)):

        nF = len(Floes)
        Floes = BreakFloes(x, t[it], Floes, Spec)
        if len(Floes) == nF:
            _ = Spec.waves(x, t[it], Floes)  # over the whole domain
            for iF in range(len(Floes)):
                Floes[iF].Eel_calc(wave, t[it])
            PlotFloes(x, t[it], Floes, wave)

    FL_temp = []
    for floe in Floes:
        FL_temp.append(floe.L)
    FL[iL] = FL_temp

if reset and (not growing):
    PlotLengths(tw, FL, wave, floe1.x0)
    PlotFSD(FL, wvlength)
