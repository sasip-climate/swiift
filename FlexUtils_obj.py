#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:23:52 2022

@author: auclaije
"""
import matplotlib.pyplot as plt
import numpy as np
import config
from pars import E, v, rho_w, g, strainCrit


def PlotFloes(x, t, Floes, wave, *args):
    hfac = 0
    nFloes = len(Floes)

    if wave.type == 'WaveSpec':
        fig, hax = wave.plot(x)
        n0 = wave.calcHs()
        n = n0
        wvstring = f'Hs_{wave.Hs:4.2f}_wl_{wave.wlp:2.0f}'
        wvtstring = f'$H_s$: {wave.Hs:4.2f}m'
    else:
        fig, hax = wave.plot(x, t, floes=Floes)
        n0 = wave.n0
        n = wave.amp(t)
        wvstring = f'n_{wave.n0:0.3}_l_{wave.wl:2.0f}'
        wvtstring = f'$\eta_0$: {n:0.3f}m'

    Eel = (nFloes - 1) * Floes[0].k
    for floe in Floes:
        if wave.type == 'WaveSpec':
            wvf = wave.calc_waves(floe.xF)
        else:
            wvf = wave.waves(floe.xF, t, amp=floe.a0, phi=floe.phi0, floes=[floe])
        _, _ = floe.plot(x, t, wvf, fig, hax)
        Eel += floe.Eel
        if floe.hw > hfac:
            hfac = floe.hw

    Hfac = 1.5 * hfac + n0
    _ = hax.axis([x[0], x[-1], -Hfac, Hfac])

    if Eel == 0:
        E_string = '0'
    else:
        oom = np.floor(np.log10(Eel))
        E_string = str(round(Eel * 10**(2 - oom)) / 100) + 'x10$^{' + str(int(oom)) + '}$'

    tstr = f'Time: {t:.2f}s - ' + wvtstring + ' - Energy: ' + E_string + 'Jm$^{-2}$'

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
        itlab = ''
        Explab = 'Floes'
        for arg in len(args):
            if type(arg) is str:
                Explab = arg
            elif isinstance(arg, (int, np.int32, np.int64)):
                itlab = f'_t_{arg:03}'
        root = (f'{Explab}_{wvstring}_'
                f'h_{Floes[0].h:3}_L_{round(Floes[-1].xF[-1]-Floes[0].x0):02}{itlab}')

        plt.savefig(config.FigsDirFloes + root + '.png')
        plt.close(fig)
    else:
        plt.show()


def PlotFracE(floe, Eel_floes, x_frac):
    fig, hax = plt.subplots()
    x = floe.xF[1:-1] - floe.x0

    if type(Eel_floes) is list:
        nF = len(Eel_floes[1])
        Etemp = np.zeros((nF, 2))
        for iF in range(nF):
            Etemp[iF, :] = np.array(Eel_floes[1][iF])
        Eel_floes = Etemp

    # Left floe
    left, = hax.semilogy(x, Eel_floes[:, 0], 'b')
    # Right floe
    right, = hax.semilogy(x, Eel_floes[:, 1], 'r')
    # Combined with fracture energy
    Etot = Eel_floes[:, 0] + Eel_floes[:, 1] + floe.k
    imin = np.argmin(Etot)
    tot, = hax.semilogy(x, Etot, ':m', linewidth=3)
    # Initial floe energy
    init, = hax.semilogy(x[[1, -1]], floe.Eel * np.array([1, 1]))
    # Fracture location
    minE = min(min(Eel_floes[:, 0]), min(Eel_floes[:, 1]))
    maxE = max(max(Eel_floes[:, 0]), max(Eel_floes[:, 1]))
    hax.semilogy(x[imin] * np.ones(2), [minE, maxE], 'k', linewidth=1)

    hax.set(ylabel='Elastic Energy (J/m$^2$)', xlabel='Along floe distance (m)')

    # Strain if available
    if hasattr(floe, 'strain'):
        hax2 = hax.twinx()
        hax2.plot(x, floe.strain[1:-1], 'g')
        hax2.set_ylabel('Elastic strain (m$^{-1}$)', color='green')

    hax.legend([init, left, right, tot],
               ['Initial Floe', 'Left Floe', 'Right Floe', 'Fractured Total'],
               loc='best')


def PlotSum(t, y, **kwargs):
    DoSave = False
    DoLeg = False
    multiFrac = 1
    for key, value in kwargs.items():
        if key == 'pstr':
            pstr = value
            DoSave = True
        elif key == 'leg':
            leg = value
            DoLeg = True
        elif key == 'multiFrac':
            multiFrac = value

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

    hax.set(ylabel='Total energy (J/m$^2$)', xlabel='Time (s)')

    if DoLeg:
        hax.legend(leg)

    if DoSave:
        plt.savefig(config.FigsDirSumry + pstr + '.png')
        plt.close()

    return(fig, hax)


def BreakFloes(x, t, Floes, wave, multiFrac=1, *args):
    '''
        Searches if a fracture can occur and where would it be
    ----------
    x : np.array -> mesh of the scene
    t : float -> time of simulation
    Floes : list(Floe) -> current floes list (order from left to right)
    wave : Wave class -> wave
    multiFrac: int -> maximum number of simultaneous fractures to look for
    *args :
        Etype: string -> energy type among 'Flex' and 'Disp'

    Returns
    -------
    Floes : list(Floe) -> updated list of floes, still in order from left to right
    '''

    EType = args[0] if len(args) > 0 else 'Flex'
    Spec = True if wave.type == 'WaveSpec' else False

    Broke = True
    nFrac = 0

    while Broke:

        Broke = False

        # Computes Elastic Energy and break Floes
        NewFloes = Floes.copy()
        Offset = 0
        Etot = 0
        for iF in range(len(Floes)):
            if Spec:
                wvf = wave.calc_waves(Floes[iF].xF)
            else:
                wvf = wave.waves(Floes[iF].xF, t, amp=Floes[iF].a0,
                                 phi=Floes[iF].phi0, floes=[Floes[iF]])
            Eel1 = Floes[iF].calc_Eel(EType=EType, wvf=wvf)

            # Check if it is worth looking for fractures
            maxFrac = Floes[iF].Eel / Floes[iF].k
            if maxFrac > 1:

                # Don't search for two fractures if k < Eel <2*k for instance
                maxFrac = min(int(maxFrac), multiFrac)
                xFracs, floes, Etot_floes, E_floes = Floes[iF].FindE_min(maxFrac, wave, t, EType)
                if Etot_floes < Eel1:
                    Broke = True
                    nFrac += len(xFracs)
                    NewFloes[iF + Offset] = floes[0]
                    for k in range(1, len(xFracs) + 1):
                        Offset += 1
                        NewFloes.insert(iF + Offset, floes[k])
                    Etot += Etot_floes
                else:
                    Etot += Eel1
            else:
                Etot += Eel1

        if Broke:
            Floes = NewFloes
            PlotFloes(x, t, Floes, wave)
        else:
            for floe in Floes:
                if floe.Eel > 10 * floe.k:
                    xf, _, _, E_lists = floe.FindE_min(1, wave, t, EType)
                    PlotFracE(floe, E_lists, xf)
            break

    return Floes


def BreakFloesStrain(x, t, Floes, wave, *args):
    ''' Use of a breaking parametrization based on strain (cf Dumont2011)
    Inputs/Outputs: same as BreakFloes
    '''

    EType = args[0] if len(args) > 0 else 'Flex'
    Spec = True if wave.type == 'WaveSpec' else False

    # Note: in the code, the nergy is computed with calc_Eel since it also computed displacement
    Broke = True
    nFrac = 0
    while Broke:

        Broke = False
        NewFloes = Floes.copy()
        Offset = 0
        for iF in range(len(Floes)):
            # Computes the wave amplitude and information along the floe
            a_vec = wave.amp_att(Floes[iF].xF, Floes[iF].a0, [Floes[iF]])
            phi0 = Floes[iF].phi0
            kw = Floes[iF].kw

            # Compute energy to compute displacement
            if Spec:
                wvf = wave.calc_waves(Floes[iF].xF)
            else:
                wvf = wave.waves(Floes[iF].xF, t, amp=Floes[iF].a0,
                                 phi=Floes[iF].phi0, floes=[Floes[iF]])
            _ = Floes[iF].calc_Eel(EType=EType, wvf=wvf)

            # Computes the strain at top and bottom edges of the floe
            Floes[iF].calc_strain()
            strain = np.abs(Floes[iF].strain)

            if np.max(strain) > strainCrit:
                Broke = True

                # Since time discretization causes intervals where strain is too high
                # we dont want to break the floe at each point of the interval
                iFracs = []
                lengthInterval = 0
                startInterval = -1
                for ix in range(len(Floes[iF].xF)):
                    if strain[ix] > strainCrit:
                        if lengthInterval == 0:
                            startInterval = ix
                        lengthInterval += 1
                    else:
                        if lengthInterval > 0:
                            iFracs.append(startInterval + (lengthInterval - 1) // 2)
                            lengthInterval = 0

                # Computes positions of fracturation and resulting floes
                xFracs = [Floes[iF].xF[i] for i in iFracs]
                createdFloes = Floes[iF].fracture(xFracs)
                nFrac += len(iFracs)

                # Set properties induced by wave and insert floes in list
                distanceFromLeft = 0
                nFloes = len(iFracs) + 1
                for iNF in range(nFloes):
                    # Set properties and calculate waves
                    if Spec:
                        wvf = wave.calc_waves(createdFloes[iNF].xF)
                    else:
                        createdFloes[iNF].a0 = a_vec[0] if iNF == 0 else a_vec[iFracs[iNF - 1]]
                        createdFloes[iNF].phi0 = phi0 + kw * distanceFromLeft
                        wvf = wave.waves(createdFloes[iNF].xF, t, amp=createdFloes[iNF].a0,
                                         phi=createdFloes[iNF].phi0, floes=[createdFloes[iNF]])
                    _ = createdFloes[iNF].calc_Eel(EType=EType, wvf=wvf)

                    distanceFromLeft += createdFloes[iNF].L
                    # Insert in list
                    if iNF == 0:
                        NewFloes[iF + Offset] = createdFloes[0]
                    else:
                        Offset += 1
                        NewFloes.insert(iF + Offset, createdFloes[iNF])

        if Broke:
            Floes = NewFloes
            PlotFloes(x, t, Floes, wave)

    return Floes


def PlotLengths(t, L, **kwargs):
    nt = len(t)
    addThickness = False
    tstring = ''
    addWaves = False
    wstring = ''
    xu = 'phase'

    for key, value in kwargs.items():
        if key == 'waves':
            addWaves = True
            waves = value
            wstring = (f'$\lambda$={waves.wl}m, $\eta_0$={waves.n0}m, '
                       f"{'constant' if {waves.beta == 0} else 'growing'} waves")
        elif key == 'Spec':
            addWaves = True
            Spec = value
            wstring = (f'$\lambda$={Spec.wlp}m, $H_s$={Spec.Hs}m waves')
        elif key == 'x0':
            x0 = value
        elif key == 'h':
            addThickness = True
            hstring = f'{value}m ice'
        elif key == 'xunits':
            xu = value

    if addWaves and addThickness:
        hstring += ' with '

    if addWaves or addThickness:
        tstring = 'Floe lengths for ' + hstring + wstring

    if nt == 1:
        fig, hax = PlotLengths1(L, tstring)
        return(fig, hax)

    fig, hax = plt.subplots()
    frmt = ['x-c', 'x-b', 'x-m', 'x-r', 'x-y']
    for it in range(nt):
        tvec = [t[it], t[it]]
        hax.plot(tvec, [0, L[it][0]], frmt[0], linewidth=3)
        L0 = L[it][0]
        for iL in range(1, len(L[it])):
            hax.plot(tvec, L0 + np.array([0, L[it][iL]]), frmt[iL % 5])
            L0 = L0 + L[it][iL]

    if addWaves:
        hax.plot(t, L0 * 1.2 + L0 * 0.1 * waves.waves(x0, t * waves.T / (2 * np.pi), amp=1))

    if addThickness or addWaves:
        hax.set_title(tstring)

    hax.set(ylabel='Floe length (m)')
    if xu == 'phase':
        hax.set(xlabel='Initial phase (rad)')
        plt.xticks(np.arange(0, 2.5, 0.5) * np.pi, ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
    elif xu == 'trials':
        hax.set(xlabel='Trial number')

    return(fig, hax)


def PlotLengths1(L, *args):
    fig, hax = plt.subplots()
    nFloes = len(L[0])
    for iF in range(nFloes):
        hax.plot([0, L[0][iF]], [iF, iF])
    hax.set(xlabel='Floe Length (m)', ylabel='Floe number')
    return(fig, hax)


def PlotFSD(L, **kwargs):
    # Process input
    if type(L[0]) == list:
        Ll = []
        for l in L:
            Ll += l[:-1]  # Do not consider the last floe in the FSD
    else:
        Ll = L

    Ll = np.array(Ll)

    # Process optional inputs
    DoSave = False
    fn = ''
    wle = False
    he = False
    ne = False
    Lines = []

    L_min = np.floor(min(Ll)) - 1
    L_max = np.ceil(max(Ll)) + 1
    dL = 0.2 if L_min < 10 else 0.5

    # Process optional inputs
    for key, value in kwargs.items():
        if key == 'FileName':
            DoSave = True
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
        elif key == 'dL':
            dL = value
        elif key == 'Lines':
            Lines = value

    values, edges = np.histogram(Ll, bins=np.arange(L_min, L_max, dL))

    fac = [1]
    fac.append(1 / values.sum())
    fac.append((edges[:-1] + 0.5) / values.sum())

    ylab = ['Number', 'Frequency', 'Length-fraction']

    if wle:
        Lines.append([wl / 2, '$\lambda$/2'])
    if he:
        Lines.append([(np.pi / 4) * (E * hv**3 / (36 * (1 - v**2) * rho_w * g))**(1 / 4), '$x^*$'])
        # Lines.append([(hv * wl)**(1 / 2), '$\sqrt{h\lambda}$'])
        # if ne:
        #     Lines.append([hv * wl / (18 * n0), '$h\lambda$/18$\eta$'])

    for ifac in np.arange(len(fac)):
        fig, hax = PlotHist(edges, values * fac[ifac])
        hax.set(ylabel=ylab[ifac])
        addLines(hax, Lines)

        if DoSave:
            root = f'FSD_{ylab[ifac]}{fn}'

            plt.savefig(config.FigsDirSumry + root + '.png')
            plt.close()

    return(edges, values)


def PlotHist(edges, values):

    fig, hax = plt.subplots()
    plt.bar(edges[:-1], values, align='edge', width=edges[1] - edges[0])

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
