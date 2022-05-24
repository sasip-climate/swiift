#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:23:01 2022

@author: auclaije
"""
import numpy as np
import matplotlib.pyplot as plt
from pars import rho_w, g
from WaveUtils import calc_k


class Wave(object):
    """ Wave field for floe breaking experiment
    Inputs: n_0:        wave amplitude (m)
            wvlength:   wave length (m)
    Optional:   beta:     time scaling factor for amplitude (1/s)
                phi:      initial wave phase (rad)
    """

    def __init__(self, n_0, wvlength, **kwargs):
        self.type = 'Wave'
        self.n0 = n_0
        self.E0 = (n_0**2) / 2

        self.wl = wvlength

        # Wave Parameters
        self.k = 2 * np.pi / wvlength
        self.omega = np.sqrt(g * self.k)
        self.T = 2 * np.pi / self.omega

        self.beta = 0
        self.phi = 0

        for key, value in kwargs.items():
            if key == 'beta':
                self.beta = value
            elif key == 'phi':
                if np.isnan(value):
                    self.phi = 2 * np.pi * np.random.random()
                else:
                    self.phi = value

    def __repr__(self):
        string = f'Wave object ({self.n0}, {self.wl}'
        if self.beta > 0:
            string += f', {self.beta})'
        else:
            string += ')'
        return(string)

    def __str__(self):
        string = f'Wave object of wave height {self.n0}m and wavelength {self.wl}m'
        if self.beta > 0:
            string += f' growing over a time scale of {1/self.beta}s'
        return(string)

    def amp(self, t):
        if self.beta == 0:
            output = self.n0
        else:
            output = self.n0 * (1 - np.e**(-self.beta * t))
        return(output)

    def calc_phase(self, x, t, **kwargs):
        floes = []
        phi0 = self.phi
        calc_phi0 = True
        Spec = False

        for key, value in kwargs.items():
            if key == 'phi':
                if np.isscalar(value):
                    phi0 = value
                    calc_phi0 = False
                elif len(value) == len(x):
                    return(value)
            elif key == 'floes':
                floes = value
            elif key == 'iF':
                iF = value
                Spec = True

        # array of phase over the domain
        phase = self.k * x - self.omega * t + phi0

        # computes the phase along the floes from left to right
        for floe in floes:
            if hasattr(floe, 'kw'):
                k = floe.kw
            elif hasattr(floe, 'kv') and Spec:
                k = floe.kv[iF]
            else:
                k = calc_k(self.omega / (2 * np.pi), floe.h, DispType=floe.DispType)

            # Phase under the floe
            if calc_phi0:
                # gets two last phase values before entering the floe
                ind = np.where(x <= floe.x0)[0][-2:]
                phip = phase[ind]
                xp = x[ind]
                # computes the phase at point x0, where the floe starts
                phi0 = phip[0] + (floe.x0 - xp[0]) * (phip[1] - phip[0]) / (xp[1] - xp[0])
                floe.phi0 = phi0

            ind = (x >= floe.x0) * (x <= floe.x0 + floe.L)
            phase[ind] = phi0 + k * (x[ind] - floe.x0)
            # Remaining domain (assume water, other floes will be looped over)
            ind = x > floe.x0 + floe.L
            phase[ind] = phi0 + k * floe.L + self.k * (x[ind] - floe.x0 - floe.L)

        return phase

    def waves(self, x, t, **kwargs):
        '''Computes the wave field over the domain,
           taking into account the different dispersion for water and ice'''
        amp = []
        phase = []
        floes = []

        for key, value in kwargs.items():
            if key == 'amp':
                amp = value
            elif key == 'phi':
                phase = value
            elif key == 'floes':
                floes = value

        if np.isscalar(phase) or len(phase) == 0:
            phase = self.calc_phase(x, t, **kwargs)

        if np.isscalar(amp) and len(floes) > 0:
            amp = self.amp_att(x, amp, floes)
        elif len(amp) == 0 and len(floes) > 0:
            amp = self.amp_att(x, self.amp(t), floes)
        elif len(amp) == 0:
            amp = self.amp(t)
        return amp * np.sin(phase)

    def mslf(self, x0, L, t):
        A = self.amp(t) / (self.k * L)
        P1 = np.cos(self.k * x0 - self.omega * t + self.phi)
        P2 = np.cos(self.k * (x0 + L) - self.omega * t + self.phi)
        return(A * (P1 - P2))

    def plot(self, x, t, **kwargs):
        # Sea surface plot
        (fig, hax) = plt.subplots()
        if len(kwargs) > 0:
            hax.plot(x, self.waves(x, t, **kwargs))
        else:
            hax.plot(x, self.waves(x, t))

        return(fig, hax)

    def amp_att(self, x, a0, floes):
        # Note:  for a single wave, E = (1/8) * rho_w * g * H^2 (laing1998guide)
        # Note2: attenuation is calculated using Sutherland et al, 2019
        #        with free parameter \epsilon \Delta_0 = 0.5 from BicWin Data
        def alpha(h, k):
            return (1 / 2) * h * k**2

        def a_att(x, h, a0, k):
            E0 = (1 / 8) * rho_w * g * (2 * a0)**2
            Ex = E0 * np.exp(-alpha(h, k) * x)
            ax = np.sqrt(8 * Ex / (rho_w * g)) / 2
            return ax

        # Initialize wave amplitude. No attenuation in open waters
        # Note: no attenuation at the first point of a floe
        ax = np.zeros_like(x, dtype=float)
        ax[x <= floes[0].x0] = a0

        nF = len(floes)
        for iF in range(nF):
            floes[iF].a0 = a0
            pFloe = (x >= floes[iF].x0) * (x <= floes[iF].x0 + floes[iF].L)
            xvec = np.append([floes[iF].x0], np.append(x[pFloe], floes[iF].x0 + floes[iF].L))
            avec = a_att(xvec - floes[iF].x0, floes[iF].h, a0, self.k)
            ax[pFloe] = avec[1:-1]
            ax[x > floes[iF].xF[-1]] = avec[-1]
            a0 = avec[-1]

        return(ax)
