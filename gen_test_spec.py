"""Simple script to generate test targets, spectral version"""

import itertools
import numpy as np
import config
from tqdm import tqdm
from FlexUtils_obj import BreakFloes, BreakFloesStrain
from WaveSpecDef import WaveSpec
from IceDef import Floe
from treeForFrac import getFractureHistory, InitHistory

multiFrac = False
FractureCriterion = 'Energy'

# Test parameters
wind_speeds = (4.3, 5.6)
thicknesses = (.7, 1.9)
space_steps = (0.5, .2)
phases = ([0.4], [1.1])

DispType = 'Open'
dist_type_str = {"Open": "free_surf", "ML": "mass_load"}
folder = f"tests/target/spec_{dist_type_str[DispType]}"
print(folder)

# Ice parameters
x0 = 50
L0 = 300
L = L0
EType = 'Flex'
# Initialize ice floe object


for comb in itertools.product(wind_speeds,
                              thicknesses,
                              space_steps,
                              phases):
    u, h, dx, phi = comb
    file_root = (f"wsp_{u:1.1f}_"
                 f"thk_{h:1.1f}_"
                 f"dlx_{dx:1.1f}_"
                 f"phs_{phi[0]:1.1f}")

    floe1 = Floe(h, x0, L, DispType=DispType, dx=dx)

    # Wave Parameters
    u = 5  # Wind speed (m/s)
    # Initialize wave object
    Spec = WaveSpec(u=u)

    # calculate wave properties in ice
    Spec.checkSpec(floe1)
    ki = floe1.kw
    floe1.setWPars(Spec)

    xi = 1 / floe1.alpha
    if L > 5 * xi[Spec.f == Spec.fp]:
        print("Warning: Floe is more than 5x "
              "the attenuation length of the peak")

    # Initial setup
    x = np.arange(x0 + L + 2)
    xProp = 4 / floe1.alpha
    xProp[xProp > L] = L
    tProp = (2 * x0 / Spec.cgw + xProp / floe1.cg)
    tPropMax = max(tProp)
    tSpecM = max(tProp[Spec.Ei > 0.1 * max(Spec.Ei)])

    Floes = [floe1]
    # Visualize energy propagation in the domain

    FL = [0]

    dt = dx / (Spec.fp * Spec.wlp)
    t = np.arange(0, 2 * tSpecM + 2 / Spec.f[0], Spec.Tp / 20)

    # print(f'Note: x* = {calc_xstar(floe1)}, wlm = {2*np.pi/Spec.k[-1]/2}, '
    #       f'wlp = {2*np.pi/Spec.kp/2}, wlM = {2*np.pi/Spec.k[0]/2}')
    # lab = f'Exp_{iL:02}_E_{EType}_
    # {Spec.SpecType}_F_{FractureCriterion}_L0_{L0}'
    # LoopName = f'{lab}_h_{h:3.1f}m_Hs_{Spec.Hs:04.1f}m.txt'
    # DataPath = config.DataTempDir + LoopName
    # FracHistPath = DataPath[:-4] + '_History.txt'
    # if path.isfile(DataPath):
    #     print(f'Reading existing data for loop {iL:02}')
    #     FL[iL] = list(np.loadtxt(DataPath))

    # Change the phases of each wave
    if len(Spec.f) == 1:
        Spec.phi = np.array([phi])
    Spec.setWaves()
    # Reset the initial floe, history and domain
    if floe1.L > L0:
        L = L0
        floe1 = floe1.fracture(x0 + L0)[0]
        x = np.arange(x0 + L0 + 2)
    Floes = [floe1]
    InitHistory(floe1, t[0])

    # tqdmlab = f'Time Loop {iL:02}' if repeats > 1 else 'Time Loop'
    for it in tqdm(range(len(t)), desc=file_root):
        nF = len(Floes)

        Spec.calcExt(x, t[it], Floes)
        Spec.set_phases(x, t[it], Floes)
        if FractureCriterion == 'Energy':
            Floes = BreakFloes(x, t[it], Floes, Spec, multiFrac, EType)
        elif FractureCriterion == 'Strain':
            Floes = BreakFloesStrain(x, t[it], Floes, Spec)
        else:
            raise ValueError('Non existing fracturation criterion')

        if Floes[-1].x0 > 0.6 * L + x0:
            # Update floes in history
            # NOTE: It also update the actual floe's length
            getFractureHistory().modifyLengthDomain(L / 2)
            # Update floe resolution and matrix
            nx = max(int(np.ceil(Floes[-1].L)), 100)
            Floes[-1].xF = Floes[-1].x0 + np.linspace(0, Floes[-1].L, nx)
            Floes[-1].initMatrix()
            # And update domain
            x = np.arange(0, x[-1] + L / 2 + 1, 1)
            L *= 1.5
            print('+', end='')

    # FL_temp = []
    # for floe in Floes:
    #     FL_temp.append(floe.L)
    # np.savetxt(DataPath, FL_temp)
    # FL[iL] = FL_temp
    lengths = [floe.L for floe in Floes]
    np.savetxt(f"{folder}/{file_root}.csv", lengths)

    n0 = Spec.calcHs()
