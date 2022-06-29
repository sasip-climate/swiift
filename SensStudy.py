#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:52:15 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from MultipleFloes1D_func import MF1D
from FlexUtils_obj import PlotHist, addLines
from pars import E, v, rho_w, g

growing = [True, False]
reset = True

# Wave Parameters
n_0 = [0.1, 0.2, 0.3]
wvlength = [5, 10, 15, 20, 25]

# Ice parameters
h = [0.5, 0.75, 1, 1.25, 1.5]
x0 = 10
L = 1000
DispType = 'Open'
ETypes = ['Disp', 'Flex']

# Check if variables already exist and don't initialize them if they do to keep data
if 'Edges' not in locals():
    Edges = {}
    Values = {}
    FloeLengths = {}
count = [0, 0, 0, 0, 0]
max_count = [len(growing), len(n_0), len(wvlength), len(h), len(ETypes)]

for gr in growing:
    count[0] += 1
    count[1] = 0
    for n0 in n_0:
        count[1] += 1
        count[2] = 0
        for wl in wvlength:
            count[2] += 1
            count[3] = 0
            for hv in h:
                count[3] += 1
                count[4] = 0
                for EType in ETypes:
                    count[4] += 1
                    key = (gr, n0, wl, hv, EType)
                    ID = (f"{'growing' if gr else 'constant'}, {n0}m waves "
                          f"of {wl}m wavelength, with {hv:03.2f}m thick ice and {EType} energy.")

                    print(f'{count} of {max_count}:')
                    if key in Edges:
                        print('Already calculated', ID)
                    else:
                        print('Launching', ID)
                        if reset:
                            FloeLengths[key], Edges[key], Values[key] = \
                                MF1D(growing=gr, n_0=n0, wvlength=wl, h=hv, L=L, EType=EType, reset=reset)
                        else:
                            FloeLengths[key] = \
                                MF1D(growing=gr, n_0=n0, wvlength=wl, h=hv, L=L, EType=EType, reset=reset)

for gr in growing:
    for n0 in n_0:
        for wl in wvlength:
            for hv in h:
                for EType in ETypes:
                    edges = Edges[gr, n0, wl, hv, EType]
                    values = Values[gr, n0, wl, hv, EType]
                    ind = edges <= wl

                    if sum(ind) != len(edges):
                        edges = edges[ind]
                        values = values[ind[:-1]][:-1]

                    # Check if there were no floes smaller than a wavelength, ie nothing interesting
                    if values.sum() == 0:
                        continue

                    fac = [1]
                    fac.append(1 / values.sum())
                    fac.append((edges[:-1] + 0.5) / values.sum())

                    ylab = ['Number', 'Frequency', 'Length-fraction']

                    Lines = [[wl / 2, '$\lambda$/2'],
                             [(hv * wl)**(1 / 2), '$\sqrt{h\lambda}$'],
                             [(np.pi / 4) * (E * hv**3 / (36 * (1 - v**2) * rho_w * g))**(1 / 4), '$x^*$'],
                             [hv * wl / (18 * n0), '$h\lambda$/18$\eta$']]

                    for ifac in np.arange(len(fac)):
                        fig, hax = PlotHist(edges, values * fac[ifac])
                        hax.set(ylabel=ylab[ifac])
                        hax.set_title(f'Distribution for $\lambda$={wl}m, $\eta_0$={n0}m, h={hv}m, '
                                      f"{'growing' if gr else 'constant'} waves")
                        addLines(hax, Lines)

                        root = (f'FSD_{ylab[ifac]}_'
                                f"{'growing' if gr else 'constant'}_waves"
                                f'_{DispType}_n_{n0:3}_l_{wl:2}_'
                                f'h_{hv:3.1f}_L0_{L:04}_E_{EType}')
                        plt.savefig(config.FigsDirSumry + root + '.png', dpi=150)
