#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:52:15 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
from MultipleFloes1D_func import MF1D
from FlexUtils_obj import PlotHist, addLines

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

Edges = {}
Values = {}
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
                    print(f'{count} of {max_count}:')
                    print(f"Working on {'growing' if gr else 'constant'}, {n0}m waves "
                          f"of {wl}m wavelength, with {hv:03.2f}m thick ice and {EType} energy.")
                    Edges[gr, n0, wl, hv], Values[gr, n0, wl, hv] = \
                        MF1D(growing=gr, reset=reset, n_0=n0, wvlength=wl, h=hv, L=L, EType=EType)

for gr in growing:
    for n0 in n_0:
        for wl in wvlength:
            for hv in h:
                for EType in ETypes:
                    edges = Edges[gr, n0, wl, hv]
                    values = Values[gr, n0, wl, hv]
                    ind = edges <= wl
                    if sum(ind) != len(edges):
                        edges = edges[ind]
                        values = values[ind[:-1]][:-1]

                    fac = [1]
                    fac.append(1 / values.sum())
                    fac.append((edges[:-1] + 0.5) / values.sum())

                    ylab = ['Number', 'Frequency', 'Length-fraction']

                    Lines = [[wl / 2, '$\lambda$/2'],
                             [hv * wl / 4, '$h\lambda$/4'],
                             [hv * wl / (18 * n0), '$h\lambda$/18$\eta$']]

                    for ifac in np.arange(len(fac)):
                        fig, hax = PlotHist(edges, values * fac[ifac])
                        hax.set(ylabel=ylab[ifac])
                        hax.set_title(f'Distribution for $\lambda$={wl}m, $\eta_0$={n0}m, h={hv}m, '
                                      f"{'growing' if gr else 'constant'} waves")
                        addLines(hax, Lines)

                        root = (f'FSD_{ylab[ifac]}_'
                                f"{'growing' if gr else 'constant'} waves"
                                f'_{DispType}_n_{n0:3}_l_{wl:2}_'
                                f'h_{hv:3.1f}_L0_{L:02}')
                        plt.savefig('FigsSum/' + root + '.png')
