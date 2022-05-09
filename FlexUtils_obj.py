#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:23:52 2022

@author: auclaije
"""
import matplotlib.pyplot as plt
import numpy as np
from pars import E, v, rho_w, g


def PlotFloes(x, t, Floes, wave, *args):
    fig, hax = wave.Plot(x, t, floes=Floes)
    hfac = 0
    nFloes = len(Floes)

    Eel = (nFloes - 1) * Floes[0].k
    for floe in Floes:
        _, _ = floe.Plot(x, t, wave, fig, hax)
        Eel += floe.Eel
        if floe.hw > hfac:
            hfac = floe.hw

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


def PlotSum(t, y, **kwargs):
    DoSave = False
    DoLeg = False
    for key, value in kwargs.items():
        if key == 'pstr':
            pstr = value
            DoSave = True
        elif key == 'leg':
            leg = value
            DoLeg = True

    fmt = ['x-r', 'x--m', '^:b']
    fig, hax = plt.subplots()
    for iF in np.arange(y.shape[1]):
        if y.shape[1] > 1:
            if max(y[:, 1]) < 10 * max(y[:, 0]):
                hax.plot(t, y[:, iF], fmt[iF])
            else:
                hax.semilogy(t, y[:, iF], fmt[iF])
        else:
            hax.plot(t, y[:, iF], fmt[iF])

    hax.set(ylabel='Elastic Energy (J/m$^2$)', xlabel='Time (s)')

    if DoLeg:
        hax.legend(leg)

    if DoSave:
        plt.savefig('Figs/' + pstr + '.png')

    return(fig, hax)


def BreakFloes(x, t, Floes, wave, *args):
    if len(args) > 0:
        EType = args[0]
    else:
        EType = 'Disp'

    Broke = True
    nFrac = 0
    while Broke:

        Broke = False

        # Calculate Elastic Energy and break Floes
        NewFloes = Floes.copy()
        Offset = 0
        Etot = 0
        for iF in range(len(Floes)):
            Eel1 = Floes[iF].calc_Eel(wave=wave, t=t, EType=EType)

            # Check if it is worth looking for fractures
            if Floes[iF].Eel > Floes[iF].k:
                x_frac, floe1, floe2, EelF, _ = Floes[iF].FindE_min(wave, t, EType)
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
            Floes = NewFloes
        else:
            for floe in Floes:
                if floe.Eel > 10 * floe.k:
                    xf, _, _, _, Emat = floe.FindE_min(wave, t)
                    PlotFracE(floe, Emat, xf)
            break

    return Floes


def PlotLengths(t, L, **kwargs):
    nt = len(t)
    addThickness = False
    tstring = ''
    addWaves = False
    wstring = ''

    for key, value in kwargs.items():
        if key == 'waves':
            addWaves = True
            waves = value
            wstring = (f'$\lambda$={waves.wl}m, $\eta_0$={waves.n0}m, '
                       f"{'constant' if {waves.beta == 0} else 'growing'} waves")
        elif key == 'x0':
            x0 = value
        elif key == 'h':
            addThickness = True
            hstring = f'{value}m ice'

    fig, hax = plt.subplots()
    frmt = ['x-c', 'x-b', 'x-m', 'x-r', 'x-y']
    for it in range(nt):
        tvec = [t[it], t[it]]
        hax.plot(tvec, [0, L[it][0]], frmt[0])
        L0 = L[it][0]
        for iL in range(1, len(L[it])):
            hax.plot(tvec, L0 + np.array([0, L[it][iL]]), frmt[iL % 5])
            L0 = L0 + L[it][iL]

    if addWaves:
        hax.plot(t, L0 * 1.2 + L0 * 0.1 * waves.waves(x0, t * waves.T / (2 * np.pi), amp=1))
        if addThickness:
            hstring += ' with '

    if addThickness or addWaves:
        tstring = 'Floe lengths for ' + hstring + wstring
        hax.set_title(tstring)
    hax.set(xlabel='Initial phase (rad)', ylabel='Floe length (m)')
    plt.xticks(np.arange(0, 2.5, 0.5) * np.pi, ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

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
    fn = ''
    wle = False
    he = False
    ne = False

    # Process optional inputs
    for key, value in kwargs.items():
        if key == 'DoSave':
            DoSave = value
        elif key == 'FileName':
            fn = value
        elif key == 'h':
            hv = value
            he = True
        elif key == 'wl':
            wl = value
            wle = True
        elif key == 'n0':
            n0 = value
            ne = True

    values, edges = np.histogram(Ll, bins=np.arange(1, round(max(Ll)) + 1))

    fac = [1]
    fac.append(1 / values.sum())
    fac.append((edges[:-1] + 0.5) / values.sum())

    ylab = ['Number', 'Frequency', 'Length-fraction']

    if wle:
        Lines = [[wl / 2, '$\lambda$/2']]
        if he:
            Lines.append([(hv * wl)**(1 / 2), '$\sqrt{h\lambda}$'])
            Lines.append([(np.pi/4)*(E*hv**3/(36*(1-v**2)*rho_w*g))**(1/4), '$x^*$'])
            if ne:
                Lines.append([hv * wl / (18 * n0), '$h\lambda$/18$\eta$'])

    for ifac in np.arange(len(fac)):
        fig, hax = PlotHist(edges, values * fac[ifac])
        hax.set(ylabel=ylab[ifac])
        addLines(hax, Lines)

        if DoSave:
            root = f'FSD_{ylab[ifac]}{fn}'

            plt.savefig('FigsSum/' + root + '.png')

    return(edges, values)


def PlotHist(edges, values):

    fig, hax = plt.subplots()
    plt.bar(edges[:-1], values, align='edge', width=1)

    hax.set(xlabel='Floe length (m)')

    return(fig, hax)


def addLines(hax, Lines):

    ylims = hax.get_ylim()
    xlims = hax.get_xlim()
    xoffset = 0.02 * (xlims[1] - xlims[0])

    colors = ['magenta', 'red', 'orange', 'yellow', 'green']
    styles = ['-', '--', '-.', ':', 'loosely dotted']
    yoffset = np.arange(1, 1 / len(Lines) - 1e-12, -1 / len(Lines)) * 0.9

    for iL in np.arange(len(Lines)):
        hax.plot(Lines[iL][0] * np.ones(2), ylims, color=colors[iL], linestyle=styles[iL])
        hax.text(Lines[iL][0] + xoffset, ylims[1] * yoffset[iL], Lines[iL][1], fontsize=20)
