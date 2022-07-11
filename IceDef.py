#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:30:37 2022

@author: auclaije
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config

# Allows faster computation of displacement
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

# For multifracturing
from itertools import combinations
# from tqdm import tqdm

from ElasticMaterials import FracToughness, Lame
from WaveUtils import calc_k, calc_cg
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

    def fracture(self, x_fracs):
        ''' Returns the list of floes induced by fractures at points x_fracs
        Input: x_fracs (list of float): list, tuple or float of fracture point(s) in (x0, x0+L)
        Output: floes (list of floes): list of resulting floes
        '''

        # Sort the fractures to keep all breaking points (including edges)
        if type(x_fracs) in {list, tuple}:
            nFrac = len(x_fracs)
            x_fracs = [self.x0] + sorted(x_fracs) + [self.x0 + self.L]
        else:  # int or float
            nFrac = 1
            x_fracs = [self.x0, x_fracs, self.x0 + self.L]
        if (not self.x0 < x_fracs[1]) or (not x_fracs[-2] < self.x0 + self.L):
            print(f'x0: {self.x0} - '
                  f'x_fracs[1]: {x_fracs[1]} - '
                  f'x_fracs[-2]: {x_fracs[-2]} - '
                  f'x0+L: {self.x0 + self.L}')
            raise ValueError("Trying to fracture outside the floe")

        # Compute lengths and resulting floes
        Lengths = [x_fracs[k + 1] - x_fracs[k] for k in range(nFrac + 1)]
        floes = [Floe(self.h, x_fracs[k], Lengths[k], DispType=self.DispType,
                      dx=min(self.dx, Lengths[k] / 100)) for k in range(nFrac + 1)]

        # Relay the wave attributes if present
        if hasattr(self, 'kw'):
            for floe in floes:
                floe.kw = self.kw
        if hasattr(self, 'cg'):
            for floe in floes:
                floe.cg = self.cg
        if hasattr(self, 'alpha'):
            for floe in floes:
                floe.alpha = self.alpha

        return floes

    def z_calc(self, msl):
        self.z = msl - self.hw + self.h / 2

    def FlexA(self):
        # Computes the np.array version of flex matrix

        x = self.xF
        I = self.I

        dx = x[1] - x[0]
        N = len(x)

        if N == 101:
            # For all floes at or under 100m, ie 101 points, use a precalculated matrix
            self.A = (E * I / dx**4) * Deriv101 + (rho_w * g) * np.eye(101)
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
            A += (rho_w * g) * np.eye(N)

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
        A += (rho_w * g) * sp.eye(N, format='csr')

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
            try:
                self.w = np.linalg.solve(self.A, b)
            except np.linalg.LinAlgError:
                errorFile = open(os.getcwd() + '/LinAlgError.txt', 'a')
                errorFile.write(f'h = {self.h}\n'
                                f'x0 = {self.x0}\n'
                                f'L = {self.L}\n'
                                f'dx = {self.dx}\n'
                                f'b = {b}\n')
                errorFile.close()
                # Solve the problem with a least squares method
                try:
                    solution = np.linalg.lstsq(self.A, b)
                    self.w = solution[0]
                    errorFile = open(os.getcwd() + '/LinAlgError.txt', 'a')
                    errorFile.write(f'rank of A = {solution[2]}\n\n')
                    errorFile.close()
                except np.linalg.LinAlgError:
                    raise ValueError("Computation of w does not converge")

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
        d2w[1:-1] = (w[:-2] - 2 * w[1:-1] + w[2:])
        d2w = d2w / (dx**2)

        return d2w

    def calc_Eel(self, **kwargs):
        EType = 'Flex'
        calc_w = False
        for key, value in kwargs.items():
            if key == 'wvf':
                wvf = value
                calc_w = True
            elif key == 'EType':
                EType = value
            else:
                raise ValueError('Unknown input wave data type')

        if calc_w:
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

    def computeEnergyIfFrac(self, iFracs, wave, t, EType, verbose=False):
        ''' Computes the resulting energy is a fracture occurs at indices iFrac
        Inputs:
            iFrac (int or list of int): points where an hypothetical fracture would occur
        Outputs:
            xFracs (list of float): points of fracture
            Eel (float): resulting elastic energy
            floes (list of Floes): list of resulting floes
            verbose (logical): True for a list of energy, false for just a sum
        '''

        if isinstance(iFracs, (int, np.int32, np.int64)):
            iFracs = [iFracs]
        else:
            iFracs = list(iFracs)
        assert min(iFracs) > 0 and max(iFracs) < len(self.xF) - 1

        Spec = (wave.type == 'WaveSpec')
        if not Spec:
            a_vec = wave.amp_att(self.xF, self.a0, [self])

        # Computes the fracture points and the resulting floes
        xFracs = [self.xF[i] for i in iFracs]
        floes = self.fracture(xFracs)

        # Set properties induces by the wave and compute elastic energies
        iFracs.append(len(self.xF) - 1)
        iFracs.insert(0, 0)
        distanceFromX0 = 0
        Eel_list = []
        Eel = 0
        nFloes = len(floes)
        for iF in range(nFloes):
            EelFloe = self.energiesMatrix[iFracs[iF], iFracs[iF + 1]]
            # Compute energie only if not already computed
            if EelFloe < 0:
                if Spec:
                    wvf = wave.calc_waves(floes[iF].xF)
                else:
                    floes[iF].a0 = a_vec[0] if iF == 0 else a_vec[iFracs[iF - 1]]
                    floes[iF].phi0 = self.phi0 + self.kw * distanceFromX0
                    wvf = wave.waves(floes[iF].xF, t, amp=floes[iF].a0,
                                     phi=floes[iF].phi0, floes=[floes[iF]])

                EelFloe = floes[iF].calc_Eel(EType=EType, wvf=wvf)
                self.energiesMatrix[iFracs[iF], iFracs[iF + 1]] = EelFloe

            if verbose:
                Eel_list.append(EelFloe)
            else:
                Eel += EelFloe

            distanceFromX0 += floes[iF].L

        if verbose:
            return xFracs, Eel_list, floes
        else:
            return xFracs, Eel, floes

    def FindE_min(self, multiFrac, wave, t, **kwargs):
        ''' Finds the minimum of energy for all fracturation possible
        Inputs:
            multiFrac (int): maximum number of simultaneous fractures
            wave, t (usual)
        Outputs:
            xFracs (list of float): points inside the floe where minimal fracture occurs
            floes (list of Floe): resulting floes
            Et_min (float): minimal total energy
            Eel_floes (list): energy of each individual floe
        '''

        EType = 'Flex'
        verbose = False

        for key, value in kwargs.items():
            if key == 'EType':
                EType = value
            elif key == 'V':
                verbose = value

        maxPosition = len(self.xF) - 1
        admissibleIndices = np.arange(start=1, stop=maxPosition, dtype=int)

        # Matrix of computed elastic energies is initialized with negative values
        self.energiesMatrix = np.full((len(self.xF), len(self.xF)), -1)

        # Arrays to compare solutions given for different number of fractures
        energyMins = np.empty(multiFrac)  # Total energy
        indicesMin = np.empty(multiFrac, dtype=object)

        e_lists = [self.Eel] * (multiFrac + 1)
        for numberFrac in range(1, multiFrac + 1):
            # List of all tuple of {numberFrac} indices where a frac will be calculated
            indicesFrac = list(combinations(admissibleIndices, numberFrac))

            # TODO: could be done a lot faster with parallelization or numpy operations
            # Array of all computed energies
            if verbose:
                e_temp = [self.computeEnergyIfFrac(iFracs, wave, t, EType, verbose=True)[1]
                          for iFracs in indicesFrac]
                          # for iFracs in tqdm(indicesFrac, desc=f'Fracture Loop {numberFrac}')]
                e_lists[numberFrac] = e_temp
                energiesTot = [sum(e_list) + numberFrac * self.k for e_list in e_lists[numberFrac]]
            else:
                energiesTot = [self.computeEnergyIfFrac(iFracs, wave, t, EType)[1] + numberFrac * self.k
                               for iFracs in indicesFrac]

            # Find min and add it array of minimums
            indMin = np.argmin(energiesTot)
            energyMins[numberFrac - 1] = energiesTot[indMin]
            indicesMin[numberFrac - 1] = indicesFrac[indMin]

        # Compute global minimum to get the fracture(s) which minimizes total energy
        globalMin = np.argmin(energyMins)
        Et_min = energyMins[globalMin]
        # TODO: Make it less expensive because no need to recompute energy
        xFracs, _, floes = \
            self.computeEnergyIfFrac(indicesMin[globalMin], wave, t, EType)

        return xFracs, floes, Et_min, e_lists

    def calc_strain(self):
        dw2 = self.calc_curv()
        self.strain = self.h * dw2 / 2

    def plot(self, x, t, wvf, *args):
        if len(args) == 2:
            fig = args[0]
            hax = args[1]
        else:
            (fig, hax) = plt.subplots()

        xv = np.array(self.xF[[0, -1]])

        # Floe plots
        msl = self.mslf_int(wvf)
        self.z_calc(msl)

        if hasattr(self, 'slope'):
            slope = self.slope
        else:
            if not hasattr(self, 'w'):
                self.calc_w(wvf)
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

    def calc_alpha(self, kw=0):
        if kw == 0:
            kw = self.kw
        return (1 / 4) * kw**2 * self.h

    def setWPars(self, wave):
        if wave.type == 'WaveSpec':
            self.kw = np.zeros_like(wave.f)
            self.cg = np.zeros_like(wave.f)
            self.kw = calc_k(wave.f, self.h, DispType=self.DispType)
            self.cg = calc_cg(self.kw, self.h, DispType=self.DispType)
        else:
            self.kw = calc_k(wave.f, self.h, DispType=self.DispType)

        self.alpha = self.calc_alpha()
