#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:23:52 2022

@author: auclaije
"""
import matplotlib.pyplot as plt
import numpy as np


def PlotFloes(x, t, Floes, wave, *args):
    fig, hax = wave.Plot(x, t, floes=Floes)
    hfac = 0
    nFloes = len(Floes)

    Eel = (nFloes - 1) * Floes[0].k
    for iF in range(nFloes):
        _, _ = Floes[iF].Plot(x, t, wave, fig, hax)
        Eel += Floes[iF].Eel
        if Floes[iF].hw > hfac:
            hfac = Floes[iF].hw

    Hfac = 1.5 * hfac + wave.n0
    _ = hax.axis([x[0], x[-1], -Hfac, Hfac])

    if Eel == 0:
        E_string = '0'
    else:
        oom = np.floor(np.log10(Eel))
        E_string = str(round(Eel * 10**(2 - oom)) / 100) + 'x10$^{' + str(int(oom)) + '}$'

    tstr = f'Time: {t:.2f}s - $\eta_0$: {wave.amp(t):0.3}m - Energy: ' + E_string + 'Jm$^{-2}$'

    hax.set_title(tstr)
    hax.set(ylabel='Height (m)', xlabel='Distance (m)')

    if nFloes > 2:
        minL = Floes[0].L
        maxL = Floes[0].L
        # Lmean = 0
        for floe in Floes:
            # Lmean += floe.L
            minL = min(minL, floe.L)
            maxL = max(maxL, floe.L)
        # Lmean = Lmean / nFloes
        # fstr = f'N: {nFloes} - L: {Lmean:.2f}'
        plt.text(x[-1] * 0.8, Hfac * 0.8, f'N: {nFloes}')
        plt.text(x[-1] * 0.8, Hfac * 0.7, f'min: {minL:.2f}m')
        plt.text(x[-1] * 0.8, Hfac * 0.6, f'max: {maxL:.0f}m')

    if len(args) > 0:
        if nFloes > 2:
            root = (f'{nFloes:3}Floes_n_{wave.n0:3}_l_{wave.wvlength:2}_'
                    'h_{Floes[0].h:3}_L_{round(Floes[-1].xF[-1]-Floes[0].x0):02}_t_{args[0]:03}')
        else:
            root = (f'OneFloe_n_{wave.n0:3}_l_{wave.wvlength:2}_'
                    'h_{Floes[0].h:3}_L_{round(Floes[-1].xF[-1]-Floes[0].x0):02}_t_{args[0]:03}')

        plt.savefig('Figs/' + root + '.png')

    plt.show()


def PlotFracE(floe, Eel_floes, x_frac):
    fig, hax = plt.subplots()
    x = floe.xF[3:-4] - floe.x0

    # Left floe
    left, = hax.semilogy(x, Eel_floes[3:-4, 0], 'b')
    # Right floe
    right, = hax.semilogy(x, Eel_floes[3:-4, 1], 'r')
    # Combined with fracture energy
    tot, = hax.semilogy(x, Eel_floes[3:-4, 0] + Eel_floes[3:-4, 1] + floe.k, ':m', linewidth=3)
    # Initial floe energy
    init, = hax.semilogy(x[[1, -1]], floe.Eel * np.array([1, 1]))
    # Fracture location
    minE = min(min(Eel_floes[3:-4, 0]), min(Eel_floes[3:-4, 1]))
    maxE = max(max(Eel_floes[3:-4, 0]), max(Eel_floes[3:-4, 1]))
    hax.semilogy(x_frac * np.ones(2) - floe.x0, [minE, maxE], 'k', linewidth=1)

    hax.set(ylabel='Elastic Energy (J/m$^2$)', xlabel='Along floe distance (m)')

    # Strain if available
    if hasattr(floe, 'strain'):
        hax2 = hax.twinx()
        hax2.plot(x, floe.strain[3:-4], 'g')
        hax2.set_ylabel('Elastic strain (m$^{-1}$)', color='green')

    hax.legend([init, left, right, tot],
               ['Initial Floe', 'Left Floe', 'Right Floe', 'Fractured Total'],
               loc='best')


def PlotSum(t, y, *args):
    fmt = ['g', 'o b', '+:r']
    fig, hax = plt.subplots()
    for iF in range(y.shape[1]):
        if max(y[:, 1]) < 10 * max(y[:, 0]):
            hax.plot(t, y[:, iF], fmt[iF])
        else:
            hax.semilogy(t, y[:, iF], fmt[iF])

    hax.set(ylabel='Elastic Energy (J/m$^2$)', xlabel='Time (s)')

    if len(args) > 0:
        plt.savefig('Figs/' + args[0] + '.png')

    return(fig, hax)


def BreakFloes(x, t, Floes, wave):
    Broke = True
    nFrac = 0
    while Broke:

        Broke = False

        # Calculate Elastic Energy and break Floes
        NewFloes = Floes.copy()
        Offset = 0
        Etot = 0
        for iF in range(len(Floes)):
            Eel1 = Floes[iF].calc_Eel(wave, t)

            # Check if it is worth looking for fractures
            if Floes[iF].Eel > Floes[iF].k:
                x_frac, floe1, floe2, EelF, _ = Floes[iF].FindE_min(wave, t)
                if EelF < Eel1:
                    Broke = True
                    nFrac += 1
                    NewFloes[iF + Offset] = floe1
                    Offset += 1
                    NewFloes.insert(iF + Offset, floe2)
                    Etot += EelF
                else:
                    Etot += Eel1
            else:
                Etot += Eel1

        if Broke:
            PlotFloes(x, t, NewFloes, wave)
            Floes = NewFloes
        else:
            for floe in Floes:
                if floe.Eel > 10 * floe.k:
                    xf, _, _, _, Emat = floe.FindE_min(wave, t)
                    PlotFracE(floe, Emat, xf)
            break

    return Floes


def PlotLengths(x, L, *args):
    nx = len(x)
    fig, hax = plt.subplots()
    frmt = ['x-c', 'x-b', 'x-m', 'x-r', 'x-y']
    for ix in range(nx):
        xvec = [x[ix], x[ix]]
        hax.plot(xvec, [0, L[ix][0]], frmt[0])
        L0 = L[ix][0]
        for iL in range(1, len(L[ix])):
            hax.plot(xvec, L0 + np.array([0, L[ix][iL]]), frmt[iL % 5])
            L0 = L0 + L[ix][iL]

    if len(args) > 0:
        hax.plot(x, L0 * 1.2 + L0 * 0.1 * args[0].waves(args[1], x, amp=1))

    hax.set(xlabel='Initial time (s)', ylabel='Floe length (m)')

    return(fig, hax)


def PlotFSD(L, **kwargs):
    # Process input
    if type(L[0]) == list:
        Ll = []
        for l in L:
            Ll += l
    else:
        Ll = L

    # Process optional inputs
    DoSave = False
    wle = False
    he = False
    ne = False
    fn = ''
    for key, value in kwargs.items():
        if key == 'h':
            h = value
            he = True
        elif key == 'wl':
            wl = value
            wle = True
        elif key == 'n':
            n = value
            ne = True
        elif key == 'DoSave':
            DoSave = value
        elif key == 'FileName':
            fn = value

    hist, bin_edges = np.histogram(Ll, np.arange(1, round(max(Ll)) + 1))

    fac = [1]
    fac.append(1 / hist.sum())
    fac.append((bin_edges[:-1] + 0.5) / hist.sum())

    ylab = ['Number', 'Frequency', 'Length-fraction']

    for ifac in np.arange(len(fac)):
        fig, hax = plt.subplots()
        plt.bar(bin_edges[:-1], hist * fac[ifac], align='edge', width=1)

        hax.set(xlabel='Floe length (m)', ylabel=ylab[ifac])

        offset = 0.02 * max(Ll)

        if wle:
            ylims = hax.get_ylim()
            hax.plot(wl * np.ones(2) / 2, ylims, color='orange')
            hax.text(wl / 2 + offset, ylims[1] * 0.9, '$\lambda$/2', fontsize=20)
            if he:
                hax.plot(h * wl * np.ones(2) / 4, ylims, 'y:')
                hax.text(h * wl / 4 + offset, ylims[1] * 0.7, '$h\lambda$/4', fontsize=20)
                if ne:
                    hax.plot(h * wl * np.ones(2) / (18 * n), ylims, 'r:')
                    hax.text(h * wl / (18 * n) + offset, ylims[1] * 0.5, '$h\lambda$/18$\eta$', fontsize=20)

        if DoSave:
            root = f'FSD_{ylab[ifac]}{fn}'

            plt.savefig('FigsSum/' + root + '.png')
