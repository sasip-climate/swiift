#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:30:37 2022

@author: auclaije
"""

import numpy as np
import matplotlib.pyplot as plt
import config

# Allows faster computation of displacement
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

from ElasticMaterials import FracToughness, Lame
from pars import g, rho_i, rho_w, E, v, K, Deriv101


class Floe(object):
    """ Floe object for floe breaking experiment
    Inputs: h:  ice thickness (m)
            x0: first point within the floe (m)
            L:  floe length (m)
    """

    def __init__(self, h, x0, L, **kwargs):

        self.h = h
        self.x0 = x0
        self.L = L
        self.dx = min(1, L / 100)
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

        self.initMatrix()

    def __repr__(self):
        return(f'Floe object ({self.h}, {self.x0:4.1f}, {self.L:4.1f})')

    def __str__(self):
        return(f'Floe object of thickness {self.h}m and length {self.L}m')

    def fracture(self, x_frac):
        L1 = x_frac - self.x0
        L2 = self.L - L1

        floe1 = Floe(self.h, self.x0, L1, DispType=self.DispType, dx=min(self.dx, L1 / 100))
        floe2 = Floe(self.h, x_frac, L2, DispType=self.DispType, dx=min(self.dx, L2 / 100))

        if hasattr(self, 'kw'):
            floe1.kw = self.kw
            floe2.kw = self.kw

        return(floe1, floe2)

    def z_calc(self, msl, t):
        self.z = msl - self.hw + self.h / 2

    def FlexA(self):
        # Computes the np.array version of flex matrix

        x = self.xF
        I = self.I

        dx = x[1] - x[0]
        N = len(x)

        if N == 101:
            # For all floes at or under 100m, ie 101 points, use a precalculated matrix
            self.A = E * I * Deriv101 / (dx**4) + rho_w * g * np.eye(101)
        else:
            # Computes the five bands of the matrix -> centered differences
            A = 6 * np.eye(N) - 4 * np.eye(N, k=1) - 4 * np.eye(N, k=-1) \
                + np.eye(N, k=2) + np.eye(N, k=-2)

            # First two rows can't use centered difference
            A[0, [0, 1, 2]] = np.array([2, -4, 2])
            A[1, [0, 1, 2, 3]] = np.array([-2, 5, -4, 1])

            # Last two rows can't use centered difference either
            A[-1, [-3, -2, -1]] = np.array([2, -4, 2])
            A[-2, [-4, -3, -2, -1]] = np.array([1, -4, 5, -2])

            # Derivation factor
            A *= E * I / (dx**4)

            # Last term of homogeneous equation
            A += rho_w * g * np.eye(N)

            self.A = A

    def FlexA_sparse(self):
        # Computes the scipy.sparse version of the flex matrix

        x = self.xF
        I = self.I
        dx = x[1] - x[0]
        N = len(x)

        # Diagonal bands
        C0 = np.ones(N)

        A = sp.spdiags([C0, -4 * C0, 6 * C0, -4 * C0, C0], [-2, -1, 0, 1, 2], N, N, format='csr')
        # First two rows can't use centered difference
        A[0, [0, 1, 2]] = np.array([2, -4, 2])
        A[1, [0, 1, 2, 3]] = np.array([-2, 5, -4, 1])

        # Last two rows can't use centered difference either
        A[-1, [-3, -2, -1]] = np.array([2, -4, 2])
        A[-2, [-4, -3, -2, -1]] = np.array([1, -4, 5, -2])

        A *= E * I / (dx**4)
        A += rho_w * g * sp.eye(N, format='csr')

        self.Asp = A

    def initMatrix(self):
        # Chooses matrix type (np.ndarray or scipy.sparse) to be more efficient
        # -> cf Overleaf CodeEfficiency internship TLILI
        N = len(self.xF)
        if N > 250:
            self.FlexA_sparse()
        elif N >= 100:
            self.FlexA()
        else:
            raise ValueError('Floe should have more points')

    def mslf_int(self, wv):
        x = self.xF
        return (wv[:-1].sum() + wv[1:].sum()) * (x[1] - x[0]) / ( 2 * (x[-1] - x[0]))

    def calc_w(self, wvf):
        b = -rho_w * g * (wvf - self.mslf_int(wvf))
        if wvf.size > 250:
            # Use of sparse properties to improve efficiency #points >> 1
            self.w = spsolve(self.Asp, b)
        else:
            self.w = np.linalg.solve(self.A, b)

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

            plt.savefig(config.FigsDirFloes + '.png')

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

    def computeEnergyIfFrac(self, iFrac, a_vec, wave, t, EType):
        '''Computes the resulting energy if a fracture occurs at position xFrac'''
        xFrac = self.xF[iFrac]
        # Creates the two resulting floes
        (floe1, floe2) = self.fracture(xFrac)

        # Sets properties induced by the wave
        floe1.a0 = self.a0
        floe1.phi0 = self.phi0
        floe2.a0 = a_vec[iFrac]
        floe2.phi0 = self.phi0 + self.kw * floe1.L

        # Computes both elastic energies
        Eel1 = floe1.calc_Eel(wave=wave, t=t, EType=EType)
        Eel2 = floe2.calc_Eel(wave=wave, t=t, EType=EType)

        return(Eel1, Eel2)

    def FindE_min(self, wave, t, *args):
        x = self.xF

        if len(args) > 0:
            EType = args[0]
        else:
            EType = 'Disp'

        a_vec = wave.amp_att(self.xF, self.a0, [self])
        nFrac = len(self.xF) - 1

        # Vectorized function to compute energies
        vect_computeEnergy = \
            np.vectorize(lambda iFrac: self.computeEnergyIfFrac(iFrac, a_vec, wave, t, EType))
        indicesFrac = np.arange(start=1, stop=nFrac, dtype=int)

        EF1, EF2 = vect_computeEnergy(indicesFrac)
        Energies = EF1 + EF2 + self.k

        # Computes the positiof of minimum energy and indice of fracture
        indMin = np.argmin(Energies)
        indFrac_min = indicesFrac[indMin]

        (floe1, floe2) = self.fracture(x[indFrac_min])

        # Sets properties induced by the wave
        floe1.a0 = self.a0
        floe1.phi0 = self.phi0
        floe2.a0 = a_vec[indMin]
        floe2.phi0 = self.phi0 + self.kw * floe1.L

        # Vector of elastic floe energies for all possible fractures
        Eel_floes = np.zeros((len(self.xF), 2))
        Eel_floes[1:-1, 0] += EF1
        Eel_floes[1:-1, 1] += EF2

        return(x[indFrac_min], floe1, floe2, Energies[indMin], Eel_floes)

    def calc_strain(self):
        dw2 = self.calc_curv()
        self.strain = self.h * dw2 / 2

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
