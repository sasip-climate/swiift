#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:01:18 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from WaveUtils import PM, Jonswap, SpecVars, calc_k
from WaveDef import Wave
from pars import g


class WaveSpec(object):
    """ Wave spectrum for floe breaking experiment
    Inputs: Hs:         wave amplitude (m)
            fp:         wave length (m)
    Optional:   beta:   time scaling factor for wave height (s)
                phi:    Initial phase of the waves, nan for random (rad)
    """

    def __init__(self, **kwargs):

        # Default spectral values know to encompass a wave spectrum well
        u = 10  # wind speed (m/s)
        Hs, Tp, fp, kp, wlp = SpecVars(u)  # Parameters for a typical spectrum
        spec = 'JONSWAP'

        beta = 0
        phi = np.nan
        df = 1.1
        x = -5
        y = 15

        f = fp * df ** np.arange(x, y)

        for key, value in kwargs.items():
            if key == 'u':
                u = value
                Hs, Tp, fp, kp, wlp = SpecVars(u)
                f = fp * df ** np.arange(x, y)
            elif key == 'Hs':
                Hs = value
            elif key == 'fp':
                fp = value
                Tp = 1 / fp
                kp = (2 * np.pi * fp)**2 / g
                wlp = 2 * np.pi / kp
            elif key == 'Tp':
                Tp = value
                fp = 1 / Tp
                kp = (2 * np.pi * fp)**2 / g
                wlp = 2 * np.pi / kp
            elif key == 'wlp':
                wlp = value
                kp = 2 * np.pi / wlp
                fp = (g * kp)**0.5 / (2 * np.pi)
                Tp = 1 / fp
            elif key == 'beta':
                beta = value
            elif key == 'phi':
                phi = value
            elif key == 'df':
                df = value
                fac = np.log(1.1) / np.log(df)
                x = -np.ceil(5 * fac)
                y = np.ceil(15 * fac) + 1
            elif key == 'spec':
                spec = value
            elif key == 'f':
                f = value

        self.type = 'WaveSpec'
        self.Hs = Hs
        self.Tp = Tp
        self.fp = fp
        self.kp = kp
        self.wlp = wlp
        self.beta = beta

        if type(f) == np.ndarray:
            self.f = f
            df_vec = np.empty_like(f)
            df_vec[0] = f[1] - f[0]
            df_vec[1:-1] = (f[2:] - f[:-2]) / 2
            df_vec[-1] = f[-1] - f[-2]
            self.nf = len(self.f)
        else:
            self.f = np.array([f])
            df_vec = np.array([1])
            self.nf = 1
        self.df = df_vec

        self.k = (2 * np.pi * self.f)**2 / g
        self.cgw = 0.5 * (g / self.k)**0.5

        if type(phi) == np.ndarray:
            self.phi = phi
        else:
            self.phi = phi * np.ones_like(self.f)

        if self.nf == 1:
            self.Ei = np.array([Hs**2 / 16])
        elif spec == 'JONSWAP':
            self.Ei = Jonswap(Hs, fp, f)
        elif spec == 'PM':
            self.Ei = PM(u, f)
        else:
            print('Unknown spectrum type')
            return

        self.setWaves()
        self.af = [0] * self.nf

    def __repr__(self):
        Hsstr = f'{self.Hs:.2f}' if self.Hs > 0.1 else f'{self.Hs:.2E}'
        Tpstr  = f'{self.Tp:.2f}' if self.Tp  > 0.1 else f'{self.Tp:.2E}'
        wlpstr  = f'{self.wlp:.0f}' if self.wlp  > 1 else f'{self.wlp:.2E}'
        string = f'Wave spectrum object (Hs: {Hsstr}m, T_p: {Tpstr}s, wl_p: {wlpstr}m'
        return(string)

    def __str__(self):
        Hsstr = f'{self.Hs:.2f}' if self.Hs > 0.1 else f'{self.Hs:.2E}'
        Tpstr  = f'{self.Tp:.2f}' if self.Tp  > 0.1 else f'{self.Tp:.2E}'
        wlpstr  = f'{self.wlp:.0f}' if self.wlp  > 1 else f'{self.wlp:.2E}'
        string = (f'Wave spectrum object of {Hsstr}m significant wave height, '
                  f'{Tpstr}s peak period and {wlpstr}m peak wavelength')
        return(string)

    def checkSpec(self, floe):
        ki = calc_k(self.f, floe.h, DispType=floe.DispType)
        ind = np.isnan(ki) + (abs(ki) > 10)
        self.nf = sum(~ind)
        self.f = self.f[~ind]
        self.df = self.df[~ind]
        self.Ei = self.Ei[~ind]
        self.cgw = self.cgw[~ind]
        floe.setWPars(self)

    def calcE(self):
        return np.sum(self.df * self.Ei)

    def calcHs(self):
        E = self.calcE()
        return 4 * np.sqrt(E)

    def setWaves(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'phi':
                self.phi = value

        self.waves = [0] * self.nf
        for iF in range(self.nf):
            f = self.f[iF]
            k = (2 * np.pi * f)**2 / g
            wl = 2 * np.pi / k
            a = (2 * self.df[iF] * self.Ei[iF]) ** 0.5
            self.waves[iF] = Wave(a, wl, beta=self.beta, phi=self.phi[iF])

    def calcExt(self, x, t, *args):
        # Calculate the wave energy for a given domain x and at a given time t
        # Check if ice info was passed and set last x or the ice
        floes = []
        if len(args) > 0:
            floes = args[0]
            calcIce = True
            x0 = floes[0].x0
        else:
            calcIce = False
            x0 = x[-1]
        self.t = t

        # Make sure floes have the proper wave information
        for floe in floes:
            if not hasattr(floe, 'kw'):
                floe.setWPars(self)

        xw = self.cgw * t  # Distance traveled by the energy for each frequency
        Ex = np.zeros([len(self.f), len(x)])
        Ex[:, 0] = self.Ei

        for iF in range(len(self.f)):
            # Before the ice, just propagate energy
            Ex[iF, x <= min([xw[iF], x0])] = self.Ei[iF]

            # If there is ice, for each floe, calculate where the energy makes it
            # and propagate and attenuated spectrum
            if calcIce:
                t_prop = floes[0].x0 / self.cgw[iF]  # Time to propagate to the ice
                last = True
                if t > t_prop:
                    for ifloe in np.arange(len(floes)):
                        floe = floes[ifloe]
                        k = floe.kw[iF]
                        cg = floe.cg[iF]

                        x_prop = cg * (t - t_prop)  # Distance traveled in the ice

                        # Waves didn't make it through the floe
                        if x_prop < floe.L:
                            ind = (x >= floe.x0) * (x <= floe.x0 + x_prop)
                            last = True
                            if sum(ind) == 0:
                                break
                        # Waves did make it through
                        else:
                            ind = (x >= floe.x0) * (x <= floe.x0 + floe.L)
                            t_prop += min([x_prop, floe.L]) / cg
                            last = False

                        # Attenuation and initial energy
                        alpha = floe.alpha[iF]
                        # First floe, only water before, no attenuation

                        if ifloe == 0:
                            E0 = Ex[iF, x <= floe.x0][-1]
                        # Any other floe, propagate the last energy calculated
                        # to the actual beginning of the new floe
                        else:
                            indL = np.where(x <= floe.x0)[0][-1]
                            dx = floe.x0 - x[indL]
                            alpha_p = (1 / 2) * floes[ifloe - 1].h * k**2
                            E0 = Ex[iF, indL] * np.exp(-alpha_p * dx)
                        Ex[iF, ind] = E0 * np.exp(-alpha * (x[ind] - floe.x0))
                        if last:
                            break

                # If waves got through all the ice
                if not last and t > t_prop:
                    indL = np.where(ind)[0][-1]  # Index of the last ice point
                    x_prop = self.cgw[iF] * (t - t_prop)  # Distance left for the energy to travel
                    ind = (x > floe.x0 + floe.L) * (x <= floe.x0 + floe.L + x_prop)
                    Ex[iF, ind] = Ex[iF, indL]

            amp = (2 * Ex[iF, :] * self.df[iF]) ** 0.5
            # Where amplitude is 0, add an exponentially decreasing tail
            # to smooth the transition at a characteristic scale of \lambda/4
            iax = amp > 0
            if iax.sum() < len(iax) and iax.sum() > 0:
                indL = np.where(iax)[0][-1]
                ampL = amp[indL]
                xL = x[indL]
                ia0 = np.invert(iax)
                amp[ia0] = ampL * np.exp( -2 * self.k[iF] * (x[ia0] - xL))

            self.af[iF] = interpolate.interp1d(x, amp, kind='quadratic')

        self.Ex = Ex
        self.x = x

    def plotSpec(self, *args):
        if len(args) > 0:
            hax = args[0]
        else:
            fig, hax = plt.subplots()

        hax.plot(self.f, self.Ei, '-+')
        hax.set(xlabel='Frequency (Hz)', ylabel='Energy (m$^2$/Hz)')

        return hax

    def plotEx(self, **kwargs):
        DoSave = False
        addTime = False
        for key, value in kwargs.items():
            if key == 'fname':
                fname = value
                DoSave = True
            elif key == 't':
                t = value
                addTime = True
            else:
                print('Unexpect input type')

        fig, hax = plt.subplots()
        x = self.x
        y = self.f.reshape([len(self.f), 1]) * np.ones_like(x)
        x = np.ones([len(self.f), 1]) * x
        c = hax.pcolor(x, y, self.Ex, shading='auto')
        hax.set(xlabel='Distance (m)', ylabel='Frequency (Hz)')
        tstring = 'Wave energy (m$^2$/Hz)'
        if addTime:
            tstring += f' at t = {t:5.2f}s'
        hax.set_title(tstring)
        fig.colorbar(c, ax=hax)

        if DoSave:
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()

    def plotMean(self, x, **kwargs):
        fig, hax = plt.subplots()

        if len(kwargs) == 0:
            wv = self.waves[np.where(self.f == self.fp)[0][0]].waves(x, self.t, amp=self.Hs / 2)
        else:
            wv = self.waves[np.where(self.f == self.fp)[0][0]].waves(x, self.t, amp=self.Hs / 2, **kwargs)

        hax.plot(x, wv)
        hax.set(xlabel='Distance (m)', ylabel='Surface elevation (m)')
        return fig, hax

    def set_phases(self, x, t, floes=[]):
        self.phif = [0] * self.nf
        for iF in range(self.nf):
            phase_i = self.waves[iF].calc_phase(x, t, floes=floes, iF=iF)
            self.phif[iF] = interpolate.interp1d(x, phase_i)

    def calc_waves(self, x):

        wvfield = np.zeros_like(x, dtype='float')
        for iF in range(self.nf):

            amp = self.af[iF](x)
            phi = self.phif[iF](x)

            wvfield += amp * np.sin(phi)
        return wvfield

    def plot(self, x, **kwargs):
        createPlot = True
        for key, value in kwargs.items():
            if key == 'hax':
                hax = value
                createPlot = False

        if createPlot:
            fig, hax = plt.subplots()

        wf = self.calc_waves(x)

        hax.plot(x, wf)

        if createPlot:
            return(fig, hax)

    def plotWMean(self, x, **kwargs):
        floes = []
        fname = []

        for key, value in kwargs.items():
            if key == 'fname':
                fname = value
            elif key == 'floes':
                floes = value

        fig, hax = self.plotMean(x, floes=floes)
        self.plot(x, hax=hax)
        hax.legend(['Sig-peak', 'Actual'], loc='upper right')
        hax.set_ylim([-self.Hs * 0.85, self.Hs * 0.85])

        if len(fname) > 0:
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()
