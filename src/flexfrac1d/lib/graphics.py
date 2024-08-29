from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

from . import numerical, physics as ph


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


def animate_displacement(
    resolution,
    experiment,
):
    # idx = 10
    floe0 = experiment.history[0].subdomains[0]
    times, floes_from_time, growth_params = zip(
        *[(k, v[0], v[1]) for k, v in experiment.history.items()]
    )
    # scales = np.geomspace(.5, 32, 13)
    # dt = 7 * .095 / scales[idx]
    dt = times[1]
    total_time = times[-1]
    # experiment = read_experiment(0, True, True)
    n_ts = np.ceil(total_time / dt).astype(int)
    vertical_range = np.sqrt(np.sum(experiment.domain.spectrum._amps**2)) * 1.1

    def animate(i, lines):
        fig.suptitle(f"t = {times[i]:.2f} s, µ_max = {growth_params[i][0].max():.2f} m")
        nfloes = len(floes_from_time[i])
        xcs = {_l.get_xdata()[0]: j + 1 for j, _l in enumerate(ax.get_lines()[1:])}
        new_xcs = []
        if nfloes > 1:
            for floe in floes_from_time[i][1:]:
                xc = floe.left_edge
                if xc in xcs:
                    ax.get_lines()[xcs[xc]].set_color("C0")
                else:
                    new_xcs.append(xc)

            for xc in new_xcs:
                line = ax.axvline(xc, ymax=0.05, lw=5, c="C1")
                lines.append(line)

        xfts, yfts = [], []
        segments = []
        for j, cf in enumerate(floes_from_time[i]):
            xft = np.linspace(0, cf.length, np.ceil(cf.length).astype(int) * 10)
            yft = ph.DisplacementHandler.from_wuf(cf, growth_params[i]).compute(xft)
            # cf.displacement(
            #     xft,
            #     domain.spectrum,
            #     (
            #         growth_params[i][0]-cf.left_edge,
            #         growth_params[i][1]
            #     ),
            #     an_sol,
            #     None,
            # )
            xfts.append(xft + cf.left_edge)
            yfts.append(yft)
            segments.append(np.vstack((xft + cf.left_edge, yft)).T)
        lc.set_segments(segments)

        segments = []
        for j, cf in enumerate(floes_from_time[i]):
            xft = np.linspace(0, cf.length, np.ceil(cf.length).astype(int) * 10)
            yft = ph.FluidSurfaceHandler.from_wuf(cf, growth_params[i]).compute(xft)
            # yft = cf.forcing(
            #     xft,
            #     domain.spectrum,
            #     (
            #         growth_params[i][0]-cf.left_edge,
            #         growth_params[i][1]
            #     ),
            # )
            xfts.append(xft + cf.left_edge)
            yfts.append(yft)
            segments.append(np.vstack((xft + cf.left_edge, yft)).T)
        lc_surf.set_segments(segments)
        return lines

    fig, ax = plt.subplots(figsize=(16, 1.6))
    ax.axis([floe0.left_edge, floe0.right_edge, -vertical_range, vertical_range])
    lines = ax.plot([], [], lw=3, c="k")
    lc = LineCollection([], color="k", lw=2)
    lc_surf = LineCollection([], color="#008aa6", lw=1)
    ax.add_collection(lc)
    ax.add_collection(lc_surf)
    ax.set_xlabel("Along-floe coordinate (m)")
    ax.set_ylabel("Wave amplitude (m)")
    ani = FuncAnimation(
        fig, animate, frames=n_ts, blit=True, fargs=(lines,), interval=dt * 1e3
    )
    return ani
    # ani.save(vtitle)
