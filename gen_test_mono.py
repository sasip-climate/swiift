"""Simple script to generate target results"""

import itertools
import numpy as np
from tqdm import tqdm

import config
from FlexUtils_obj import BreakFloes, BreakFloesStrain
from WaveUtils import calc_k, omega
from WaveDef import Wave
from IceDef import Floe
from treeForFrac import InitHistory


multiFrac = False
growing = True
if growing:
    lab = 'g'
else:
    lab = '0'

# Test parameters
amplitudes = (0.2, .6)
thicknesses = (.7, 1.9)
space_steps = (0.6, .3)
phases = ([0.4], [1.1])

DispType = 'ML'
dist_type_str = {"Open": "free_surf", "ML": "mass_load"}
folder = f"tests/target/mono_{dist_type_str[DispType]}"

# Wave Parameters
wvlength = 20

# Ice parameters
x0 = 10
L0 = 240
L = L0
EType = 'Flex'
FractureCriterion = 'Energy'  # 'Strain' or 'Energy'

print(folder)
for comb in itertools.product(amplitudes,
                              thicknesses,
                              space_steps,
                              phases):
    n_0, h, dx, phi = comb
    file_root = (f"amp_{n_0:1.1f}_"
                 f"thk_{h:1.1f}_"
                 f"dlx_{dx:1.1f}_"
                 f"phs_{phi[0]:1.1f}")
    # Initialize wave object
    if growing:
        beta = omega((2 * np.pi / wvlength)) * wvlength / (2 * np.pi * L)
        wave = Wave(n_0, wvlength, beta=beta)
        t_max = 6 / beta
    else:
        wave = Wave(n_0, wvlength)
        t_max = 2 * wave.T

    # Initialize ice floe object
    floe1 = Floe(h, x0, L, DispType=DispType, dx=dx)
    floe1.kw = calc_k(1 / wave.T, h, DispType=DispType)

    # Initial setup
    x = np.arange(2 * x0 + L)

    # Necessery, as defining filename roots calls Floes which may not exist if
    # all experiments already saved
    Floes = [floe1]

    FL = [0]

    dt = dx / (5 * wave.omega / wave.k)

    t = np.arange(0, t_max, dt)  # wave.T / 20)

    wave.phi = phi[0]
    Evec = np.zeros([len(t), 1])
    Floes = [floe1]
    InitHistory(floe1, t[0])

    _ = wave.waves(x, t[0], floes=Floes)  # assign waves over the whole domain

    for it in tqdm(range(len(t)), desc=file_root):
        _ = wave.waves(x, t[it], floes=Floes)  # over the whole domain
        nF = len(Floes)

        if FractureCriterion == 'Energy':
            Floes = BreakFloes(x, t[it], Floes, wave, multiFrac, EType)
            Evec[it] = (len(Floes) - 1) * Floes[0].k
            for floe in Floes:
                Evec[it] += floe.Eel
        elif FractureCriterion == 'Strain':
            Floes = BreakFloesStrain(x, t[it], Floes, wave)
        else:
            raise ValueError('Non existing fracturation criterion')

    lengths = [floe.L for floe in Floes]
    np.savetxt(f"{folder}/{file_root}.csv", lengths)
