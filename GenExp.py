#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:49:40 2023

@author: Jean-Pierre Auclair
"""
# Script to generate pars.py files for whole experiments, with random values

import numpy as np
import os


def GenExpRand(NRuns, Folder, Vars, Pars):

    Nchars = int(np.log10(NRuns)) + 1

    if not os.path.exists(Folder):
        os.makedirs(Folder)
    cdir = os.getcwd()
    os.chdir(Folder)

    RNG = np.random.default_rng()

    for iRun in np.arange(NRuns):
        RunN = f'{iRun:0{Nchars}}'

        # Repeats are a parameter, but never multivalued
        repeats = Pars['repeats'] if 'repeats' in Pars else 20

        # Determine the value of all variables
        E = drawVal(RNG, Vars['E']) if 'E' in Vars else 6e9
        K = drawVal(RNG, Vars['K']) if 'K' in Vars else 1e5
        h = drawVal(RNG, Vars['h']) if 'h' in Vars else 1
        u = drawVal(RNG, Vars['u']) if 'u' in Vars else 5
        phi0 = drawVal(RNG, Vars['phi0'], [21, repeats]) if 'phi0' in Vars else np.nan

        SpecType = Pars['SpecType'] if 'SpecType' in Pars else [['JONSWAP']]
        FracCrit = Pars['FracCrit'] if 'FracCrit' in Pars else ['Energy']
        tail_fac = Pars['tail_fac'] if 'tail_fac' in Pars else ['2']
        multiFrac = Pars['multiFrac'] if 'multiFrac' in Pars else [True]

        # Loop over parameters for which vars will be shared
        for ST in SpecType:
            for FC in FracCrit:
                for mF in multiFrac:
                    for tf in tail_fac:
                        ExpN = f'{ST[0]}_{FC}_mF_{mF}_tf_{tf}'
                        if iRun == 0 and not os.path.exists(ExpN):
                            os.mkdir(ExpN)
                        os.chdir(ExpN)

                        if not os.path.exists(f'{ExpN}_{RunN}'):
                            os.mkdir(f'{ExpN}_{RunN}')
                        os.chdir(f'{ExpN}_{RunN}')

                        with open('pars.py', 'w') as f:
                            f.write('#!/usr/bin/env python3\n')
                            f.write('# -*- coding: utf-8 -*-\n')
                            f.write('\n')
                            f.write('import numpy as np\n')
                            f.write('\n')
                            f.write('g = 9.8\n')
                            f.write('rho_w = 1025\n')
                            f.write('rho_i = 922.5\n')
                            f.write('\n')
                            f.write(f'E = {E}\n')
                            f.write('v = 0.3\n')
                            f.write(f'K = {K}\n')
                            f.write('\n')
                            f.write('strainCrit = 3e-5\n')
                            f.write('\n')
                            f.write(f'h = {h}\n')
                            f.write(f'u = {u}\n')
                            f.write('f = 0.25\n')
                            f.write('wl = g / (2 * np.pi * f**2)\n')
                            f.write(f"SpecType = '{ST[0]}'\n")
                            f.write(f'n = {ST[1]}\n') if 'Power' in ST[0] else f.write('n = 2\n')
                            f.write(f'tail_fac = {tf}\n')
                            f.write('n0 = 1\n')
                            if 'Mono' in ST[0]:
                                f.write(f'phi0 = np.{phi0[5,:].__repr__()}\n')
                            else:
                                f.write(f'phi0 = np.{phi0.__repr__()}\n')
                            f.write('\n')
                            f.write(f"FractureCriterion = '{FC}'\n")
                            f.write(f'multiFrac = {mF}\n')
                            f.write(f'repeats = {repeats}\n')
                            f.write('\n')
                            f.write('N = 101\n')
                            f.write('Deriv101 = 6 * np.eye(N) - 4 * np.eye(N, k=1) - 4 * np.eye(N, k=-1) + np.eye(N, k=2) + np.eye(N, k=-2)\n')
                            f.write('Deriv101[0, [0, 1, 2]] = np.array([2, -4, 2])\n')
                            f.write('Deriv101[1, [0, 1, 2, 3]] = np.array([-2, 5, -4, 1])\n')
                            f.write('Deriv101[-1, [-3, -2, -1]] = np.array([2, -4, 2])\n')
                            f.write('Deriv101[-2, [-4, -3, -2, -1]] = np.array([1, -4, 5, -2])\n')
                            f.close()

                        os.chdir(f'{cdir}/{Folder}')
    os.chdir(cdir)


def drawVal(RNG, Pars, N=None):
    if 'uniform' in Pars[2]:
        val = RNG.uniform(Pars[0], Pars[1], N)
    elif 'normal' in Pars[2]:
        val = RNG.normal(Pars[0], Pars[1], N)
    else:
        raise ValueError('Unknown distribution type')

    return(val)
