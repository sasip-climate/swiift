#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:30:37 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from ElasticMaterials import FracToughness, Lame
from pars import g, rho_i, rho_w, E, v, K


class Floe(object):
    """ Floe object for floe breaking experiment
    Inputs: h:  ice thickness (m)
            x0: first point within the floe
            L:  floe length (m)
    """

    def __init__(self, h, x0, L, **kwargs):

        self.h = h
        self.x0 = x0
        self.L = L
        self.dx = L / 100
        self.hw = rho_i / rho_w * h
        self.ha = h - self.hw
        self.I = h**3 / (12 * (1 - v**2))
        self.k = FracToughness(E, v, K)

        self.DispType = 'ElML'
        for key, value in kwargs.items():
            if key == 'DispType':
                self.DispType = value
            elif key == 'dx':
                self.dx = value

        self.xF = np.arange(x0, x0 + L + self.dx / 2, self.dx)

    def __repr__(self):
        return(f'Floe object ({self.h}, {self.x0:4.1f}, {self.L:4.1f})')

    def __str__(self):
        return(f'Floe object of thickness {self.h}m and length {self.L}m')

    def fracture(self, x_frac):
        L1 = x_frac - self.x0
        floe1 = Floe(self.h, self.x0, L1, DispType=self.DispType, dx=min(self.dx, L1 / 100))
        L2 = self.x0 + self.L - x_frac
        floe2 = Floe(self.h, x_frac, L2, DispType=self.DispType, dx=min(self.dx, L2 / 100))

        if hasattr(self, 'kw'):
            floe1.kw = self.kw
            floe2.kw = self.kw

        return(floe1, floe2)

    def z_calc(self, msl, t):
        self.z = msl - self.hw + self.h / 2

    def FlexA(self):
        x = self.xF
        I = self.I

        dx = x[1] - x[0]

        A = np.zeros((len(x), len(x)))
        # First two rows can't use centered difference
        A[0, [0, 1, 2]] = E * I * np.array([2, -4, 2]) / dx**4
        A[0, 0] += rho_w * g
        A[1, [0, 1, 2, 3]] = E * I * np.array([-2, 5, -4, 1]) / dx**4
        A[1, 1] += rho_w * g

        # Last two rows can't use centered difference either
        A[-1, [-3, -2, -1]] = E * I * np.array([2, -4, 2]) / dx**4
        A[-1, -1] += rho_w * g
        A[-2, [-4, -3, -2, -1]] = E * I * np.array([1, -4, 5, -2]) / dx**4
        A[-2, -2] += rho_w * g

        stencil = np.arange(-2, 3)
        coeffs = np.array([1, -4, 6, -4, 1])
        for i in range(2, len(A) - 2):
            A[i, stencil + i] = E * I * coeffs / dx**4
            A[i, i] += rho_w * g

        return(A)

    def mslf_int(self, wv):
        x = self.xF
        return (wv[:-1].sum() + wv[1:].sum()) * (x[1] - x[0]) / ( 2 * (x[-1] - x[0]))

    def calc_w(self, wvf):
        A = self.FlexA()
        b = -rho_w * g * (wvf - self.mslf_int(wvf))
        self.w = np.linalg.solve(A, b)

    def calc_du(self, fname=''):
        x = self.xF
        w = self.w
        dx = x[1] - x[0]

        dw = np.zeros(len(w))
        dw[0]  = (-3 * w[0]  + 4 * w[1]  - w[2])
        dw[-1] = ( 3 * w[-1] - 4 * w[-2] + w[-3])
        dw[1:-1] = (-w[:-2] + w[2:])
        dw = dw / (2 * dx)

        # Remove constant dwdx from the set, to account for rotation of the floe
        fit = np.polyfit(x, dw, 0)
        self.du = dw  - fit[0]

        if len(fname) > 0:
            fitw = np.polyfit(x, w, 1)
            fig, hax = plt.subplots(1, 2)

            hax[0].plot(x - x[0], w)
            hax[0].plot(x - x[0], x * fitw[0] + fitw[1], ':')
            hax[0].set_title(f'Deformations fit:\n {fitw[0]:0.6}x + {fitw[1]:0.6}')

            hax[1].plot(x - x[0], dw)
            hax[1].plot(x - x[0], fit * np.ones_like(x), ':')
            hax[1].set_title(f'Deformation gradient: {fit[0]:.6f}\n')

            plt.savefig(fname + '.png')

        self.slope = fit[0]
        return self.du

    def calc_curv(self):
        x = self.xF
        w = self.w
        dx = x[1] - x[0]

        d2w = np.zeros(len(w))
        d2w[0]  = (  2 * w[0]  - 5 * w[1]  + 4 * w[2]  - 1 * w[3])
        d2w[-1] = ( -2 * w[-1] + 5 * w[-2] - 4 * w[-3] + 1 * w[-4])
        d2w[1:-1] = (w[:-2] - 2 * w[1:-1] + w[2:])
        d2w = d2w / (dx**2)

        return d2w

    def calc_Eel(self, **kwargs):
        EType = 'Flex'
        calc_wvf = False
        for key, value in kwargs.items():
            if key == 't':
                t = value
            elif key == 'wave':
                wave = value
                calc_wvf = True
            elif key == 'EType':
                EType = value
            else:
                raise ValueError('Unknown input wave data type')

        if calc_wvf:
            wvf = wave.waves(self.xF, t, amp=self.a0, phi=self.phi0, floes=[self])
            self.calc_w(wvf)

        if EType == 'Disp':
            intV = self.calc_du()
            (l, u) = Lame(E, v)
            prefac = u
        elif EType == 'Flex':
            intV = self.calc_curv()
            prefac = (1 / 2) * E * self.I / self.h

        int2 = (intV[0]**2 / 2 + intV[-1]**2 / 2 + (intV[1:-1]**2).sum()) * self.dx

        self.Eel = prefac * int2

        return(self.Eel)

    def FindE_min(self, wave, t, *args):
        if len(args) > 0:
            EType = args[0]
        else:
            EType = 'Disp'

        Eel_floes = np.empty((len(self.xF), 2))

        a_vec = wave.amp_att(self.xF, self.a0, [self])

        # Can't run calculations on floes less than 4 long, so cut 3 from the end
        nFrac = len(self.xF) - 3

        Eel_min = self.Eel

        if nFrac < 4:
            x_fracm = self.xF[0]
            floe1m = self
            floe2m = []
        else:
            for iFrac in range(3, nFrac):
                x_frac = self.xF[iFrac]

                (floe1, floe2) = self.fracture(x_frac)

                floe1.a0 = self.a0
                floe1.phi0 = self.phi0
                floe2.a0 = a_vec[iFrac]
                floe2.phi0 = self.phi0 + self.kw * floe1.L

                Eel1 = floe1.calc_Eel(wave=wave, t=t, EType=EType)
                Eel2 = floe2.calc_Eel(wave=wave, t=t, EType=EType)

                Eel_floes[iFrac, 0] = Eel1
                Eel_floes[iFrac, 1] = Eel2

                Eel_f_t = Eel1 + Eel2 + self.k

                if Eel_f_t < Eel_min or iFrac == 3:
                    x_fracm = x_frac
                    Eel_min = Eel_f_t
                    floe1m = copy.deepcopy(floe1)
                    floe2m = copy.deepcopy(floe2)

        return(x_fracm, floe1m, floe2m, Eel_min, Eel_floes)

    def calc_strain(self):
        x = self.xF
        dx = x[1] - x[0]
        du = self.du

        du2 = np.zeros(len(du))
        du2[0]  = (-3 * du[0]  + 4 * du[1]  - du[2])
        du2[-1] = ( 3 * du[-1] - 4 * du[-2] + du[-3])
        du2[1:-1] = (-du[:-2] + du[2:])
        du2 = du2 / (2 * dx)

        self.strain = self.h * du2 / 2

    def Plot(self, x, t, wave, *args):
        if len(args) == 2:
            fig = args[0]
            hax = args[1]
        else:
            (fig, hax) = plt.subplots()

        xv = np.array(self.xF[[0, -1]])

        # Floe plots
        wvf = wave.waves(self.xF, t, amp=self.a0, phi=self.phi0, floes=[self])
        msl = self.mslf_int(wvf)
        self.z_calc(msl, t)

        if hasattr(self, 'slope'):
            slope = self.slope
        else:
            fitw = np.polyfit(self.xF, self.w, 1)
            slope = fitw[0]

        sfac = slope * self.L / 2
        z_vec = np.array([sfac, -sfac])

        hax.plot(xv, z_vec + msl, 'b:')
        hax.plot(xv, z_vec + self.z, 'r:')
        hax.plot(xv, z_vec + msl + self.ha, 'c:')
        hax.plot(xv, z_vec + msl - self.hw, 'k:')

        hax.plot(self.xF, -self.w + self.z, 'r')
        hax.plot(self.xF, -self.w + msl + self.ha, 'c')
        hax.plot(self.xF, -self.w + msl - self.hw, 'k')

        return(fig, hax)
