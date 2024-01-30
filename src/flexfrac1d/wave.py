#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:23:01 2022

@author: auclaije
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import ArrayLike

from .pars import rho_w, g
from .libraries.WaveUtils import calc_k
from .ice import Floe


class Wave(object):
    """Wave field for floe breaking experiment"""

    def __init__(
        self, amplitude: float, wavelength: float, phase: float = 0, beta: float = 0
    ):
        self.__amplitude = amplitude
        self.__wavelength = wavelength
        if np.isnan(phase):
            phase = 2 * np.pi * np.random.random()
        self.__phase = phase
        self.__beta = beta

    @property
    def amplitude(self):
        return self.__amplitude

    @property
    def wavelength(self):
        return self.__wavelength

    @property
    def phase(self):
        return self.__phase

    @property
    def beta(self):
        return self.__beta

    @functools.cached_property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    @functools.cached_property
    def ang_frequency(self):
        return np.sqrt(g * self.wavenumber)

    @functools.cached_property
    def period(self):
        return 2 * np.pi / self.ang_frequency

    @functools.cached_property
    def energy(self):
        return self.amplitude**2 / 2

    @property
    def type(self):
        return "Wave"

    def __repr__(self):
        n0str = (
            f"{self.amplitude:.2f}" if self.amplitude > 0.1 else f"{self.amplitude:.2E}"
        )
        tstr = f"{self.period:.2f}" if self.period > 0.1 else f"{self.period:.2E}"
        wlstr = (
            f"{self.wavelength:.0f}" if self.wavelength > 1 else f"{self.period:.2E}"
        )
        string = f"Wave object (n0: {n0str}m, T: {tstr}s, wl: {wlstr}m"
        if self.beta > 0:
            string += f", {self.beta})"
        else:
            string += ")"
        return string

    def __str__(self):
        n0str = (
            f"{self.amplitude:.2f}" if self.amplitude > 0.1 else f"{self.amplitude:.2E}"
        )
        tstr = f"{self.period:.2f}" if self.period > 0.1 else f"{self.period:.2E}"
        wlstr = (
            f"{self.wavelength:.0f}"
            if self.wavelength > 1
            else f"{self.wavelength:.2E}"
        )
        string = (
            f"Wave object of wave height {n0str}m, "
            f"period {tstr}s and wavelength {wlstr}m"
        )
        if self.beta > 0:
            string += f" growing over a time scale of {1/self.beta}s"
        return string

    def amp(self, t: float) -> float:
        """Time-dependent wave amplitude

        Allow for wave growth when the parameter beta is non-zero.

        Parameters
        ----------
        t : float
            Timestep in s

        Returns
        -------
        float
            The computed amplitude in m

        """
        if self.beta == 0:
            return self.amplitude
        return self.amplitude * (1 - np.exp(-self.beta * t))

    def calc_phase(
        self,
        x: ArrayLike | float,
        t: float,
        phase: ArrayLike | float | None = None,
        floes: List[Floe] = [],
        iF: int | None = None,
    ) -> ArrayLike:
        floes = []

        if phase is None:
            calc_phi0 = True
            phase = self.phase
        else:
            if np.isscalar(phase):
                calc_phi0 = False
            elif len(phase) == len(x):
                return phase

        if iF is not None:
            spec = True
        else:
            spec = False

        # array of phase over the domain
        phase = self.wavenumber * x - self.ang_frequency * t + phase

        # computes the phase along the floes from left to right
        for floe in floes:
            if hasattr(floe, "kw"):
                if spec:
                    k = floe.kw[iF]
                else:
                    k = floe.kw
            else:
                k = calc_k(
                    self.ang_frequency / (2 * np.pi), floe.h, DispType=floe.DispType
                )

            # Phase under the floe
            if calc_phi0:
                # gets two last phase values before entering the floe
                ind = np.where(x <= floe.x0)[0][-2:]
                phip = phase[ind]
                xp = x[ind]
                # computes the phase at point x0, where the floe starts
                phase = phip[0] + (floe.x0 - xp[0]) * (phip[1] - phip[0]) / (
                    xp[1] - xp[0]
                )
                floe.phi0 = phase

            ind = (x >= floe.x0) * (x <= floe.x0 + floe.L)
            phase[ind] = phase + k * (x[ind] - floe.x0)
            # Remaining domain (assume water, other floes will be looped over)
            ind = x > floe.x0 + floe.L
            phase[ind] = (
                phase + k * floe.L + self.wavenumber * (x[ind] - floe.x0 - floe.L)
            )

        return phase

    def waves(self, x: ArrayLike, t: float, **kwargs):
        """Computes the wave field over the domain,
        taking into account the different dispersion for water and ice"""
        amp = []
        phase = []
        floes = []

        for key, value in kwargs.items():
            if key == "amp":
                amp = value
            elif key == "phi":
                phase = value
            elif key == "floes":
                floes = value

        if np.isscalar(phase) or len(phase) == 0:
            phase = self.calc_phase(x, t, **kwargs)

        if np.isscalar(amp) and len(floes) > 0:
            amp = self.amp_att(x, amp, floes)
        elif amp == [] and len(floes) > 0:
            amp = self.amp_att(x, self.amp(t), floes)
        elif amp == []:
            amp = self.amp(t)
        return amp * np.sin(phase)

    def mslf(self, x0, L, t):
        A = self.amp(t) / (self.wavenumber * L)
        P1 = np.cos(self.wavenumber * x0 - self.ang_frequency * t + self.phase)
        P2 = np.cos(self.wavenumber * (x0 + L) - self.ang_frequency * t + self.phase)
        return A * (P1 - P2)

    def plot(self, x, t, **kwargs):
        # Sea surface plot
        (fig, hax) = plt.subplots()
        if len(kwargs) > 0:
            hax.plot(x, self.waves(x, t, **kwargs))
        else:
            hax.plot(x, self.waves(x, t))

        return fig, hax

    def amp_att(self, x: ArrayLike, a0: float, floes: List[Floe]) -> ArrayLike:
        # Note:  for a single wave,
        # E = (1/8) * rho_w * g * H^2 (laing1998guide)
        # Note2: attenuation is calculated using Sutherland et al, 2019
        #        with free parameter \epsilon \Delta_0 = 0.5 from BicWin Data

        def a_att(x, floe, a0, k):
            E0 = (1 / 8) * rho_w * g * (2 * a0) ** 2
            Ex = E0 * np.exp(-floe.calc_alpha(k) * x)
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
            xvec = np.append(
                [floes[iF].x0], np.append(x[pFloe], floes[iF].x0 + floes[iF].L)
            )
            avec = a_att(xvec - floes[iF].x0, floes[iF], a0, self.wavenumber)
            ax[pFloe] = avec[1:-1]
            ax[x > floes[iF].xF[-1]] = avec[-1]
            a0 = avec[-1]

        return ax
