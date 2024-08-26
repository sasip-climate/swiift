from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

from . import numerical


def _linspace_nums(resolution: float, floes) -> list[int]:
    return [np.ceil(floe.length / resolution).astype(int) + 1 for floe in floes]


def _surface_segments(resolution, domain, left_bound, an_sol) -> list[np.ndarray]:
    nxs = _linspace_nums(resolution, domain.subdomains)
    xfs = np.linspace(
        left_bound,
        domain.subdomains[0].left_edge,
        np.ceil((domain.subdomains[0].left_edge - left_bound) / resolution).astype(int)
        + 1,
    )
    growth_params = None if an_sol else (domain.growth_mean, domain.growth_std)
    yfs = numerical.free_surface(
        xfs,
        (domain.spectrum._amps, domain.ocean.wavenumbers, domain.spectrum._phases),
        growth_params,
    )
    segments = [np.vstack((xfs, yfs)).T] + [np.full((nx, 2), np.nan) for nx in nxs]
    for i, (nx, floe) in enumerate(zip(nxs, domain.subdomains), 1):
        segments[i][:, 0] = np.linspace(0, floe.length, nx)
        growth_params = None if an_sol else domain._pack_growth(floe)
        segments[i][:, 1] = floe.forcing(
            segments[i][:, 0], domain.spectrum, growth_params
        )
        segments[i][:, 0] += floe.left_edge
    return segments


def _dis_segments(resolution, domain, an_sol, base):
    # TODO: use `base` to offset, e.g. to the bottom or the top of the floe
    nxs = _linspace_nums(resolution, domain.subdomains)
    segments = [np.full((nx, 2), np.nan) for nx in nxs]
    for i, (nx, floe) in enumerate(zip(nxs, domain.subdomains)):
        segments[i][:, 0] = np.linspace(0, floe.length, nx)
        segments[i][:, 1] = floe.displacement(
            segments[i][:, 0], domain.spectrum, domain._pack_growth(floe), an_sol, None
        )
        segments[i][:, 0] += floe.left_edge
    return segments


def plot_displacement(
    resolution,
    domain,
    left_bound=None,
    ax=None,
    an_sol=None,
    add_surface=True,
    base=0,
    kw_dis=None,
    kw_sur=None,
):
    if kw_dis is None:
        kw_dis = {"color": "k", "lw": 3}
    displacements = LineCollection(
        _dis_segments(resolution, domain, an_sol, base), **kw_dis
    )
    if left_bound is None:
        left_bound = domain.subdomains[0].left_edge
    if ax is None:
        ax = plt.gca()
        wave_height = np.sum(domain.spectrum._amps**2) ** 0.5 * 1.1
        ax.axis(
            [left_bound, domain.subdomains[-1].right_edge, -wave_height, wave_height]
        )

    if add_surface:
        if kw_sur is None:
            kw_sur = {"color": "#008aa6", "lw": 1.5}
        surface = LineCollection(
            _surface_segments(resolution, domain, left_bound, an_sol), **kw_sur
        )
        ax.add_collection(surface)
    ax.add_collection(displacements)
    return ax
