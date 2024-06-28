import numpy as np
from .constants import PI_D4, SQR2
from .displacement import _dis_hom_coefs, _dis_par_amps
from . import numerical


def _egy_hom(floe_params, wave_params):
    """Energy from the homogen term of the displacement ODE"""
    red_num, length = floe_params
    adim = red_num * length
    adim2 = 2 * adim
    c_1, c_2, c_3, c_4 = _dis_hom_coefs(floe_params, wave_params)

    return red_num**3 * (
        +(c_1**2) * (-SQR2 * np.sin(adim2 + PI_D4) + 2 - np.exp(-adim2)) / 2
        + c_2**2 * (SQR2 * np.sin(adim2 + PI_D4) + 2 - 3 * np.exp(-adim2)) / 2
        + c_3**2 * ((SQR2 * np.cos(adim2 + PI_D4) - 2) * np.exp(-adim2) + 1) / 2
        + c_4**2 * (3 - (SQR2 * np.cos(adim2 + PI_D4) + 2) * np.exp(-adim2)) / 2
        + c_1 * c_2 * (SQR2 * np.cos(adim2 + PI_D4) - np.exp(-adim2))
        - 2
        * red_num
        * c_1
        * c_3
        * (2 * length - np.sin(adim2) / red_num)
        * np.exp(-adim)
        + 4 * c_1 * c_4 * np.exp(-adim) * np.sin(adim) ** 2
        + 4 * c_2 * c_3 * np.exp(-adim) * np.sin(adim) ** 2
        - 2
        * red_num
        * c_2
        * c_4
        * (2 * length + np.sin(adim2) / red_num)
        * np.exp(-adim)
        + c_3 * c_4 * (-1 + SQR2 * np.exp(-adim2) * np.sin(adim2 + PI_D4))
    )


def _egy_par_vals(red_num, wave_params):
    comp_amps = _dis_par_amps(red_num, wave_params)
    _, c_wavenumbers = wave_params

    comp_curvs = comp_amps * (1j * c_wavenumbers) ** 2

    return c_wavenumbers, comp_curvs


def _egy_par_pow2(floe_params, wave_params):
    """Energy contribution from individual forcings"""
    red_num, length = floe_params
    _, c_wavenumbers = wave_params
    wavenumbers = np.real(c_wavenumbers)
    attenuations = np.imag(c_wavenumbers)

    comp_wns, comp_curvs = _egy_par_vals(red_num, wave_params)
    wn_moduli, curv_moduli = map(np.abs, (comp_wns, comp_curvs))
    wn_phases, curv_phases = map(np.angle, (comp_wns, comp_curvs))

    red = np.exp(-2 * attenuations * length)

    return (
        curv_moduli**2
        @ (
            (1 - red) / attenuations
            + (
                np.sin(2 * curv_phases - wn_phases)
                - np.sin(2 * (wavenumbers * length + curv_phases) - wn_phases) * red
            )
            / wn_moduli
        )
    ) / 4


def _egy_par_m(floe_params, wave_params):
    """Energy contribution from forcing interactions"""
    red_num, length = floe_params
    _, c_wavenumbers = wave_params
    wavenumbers = np.real(c_wavenumbers)
    attenuations = np.imag(c_wavenumbers)
    _, comp_curvs = _egy_par_vals(red_num, wave_params)

    # Binomial coefficients, much quicker than itertools
    idx1, idx2 = np.triu_indices(c_wavenumbers.size, 1)

    mean_attenuations = attenuations[idx1] + attenuations[idx2]
    comp_wns = (
        wavenumbers[idx1] - wavenumbers[idx2],
        wavenumbers[idx1] + wavenumbers[idx2],
    )
    comp_wns += 1j * mean_attenuations

    curv_moduli = np.abs(comp_curvs)
    _curv_phases = np.angle(comp_curvs)
    curv_phases = (
        _curv_phases[idx1] - _curv_phases[idx2],
        _curv_phases[idx1] + _curv_phases[idx2],
    )

    def _f(comp_wns, curv_phases):
        wn_moduli = np.abs(comp_wns)
        wn_phases = np.angle(comp_wns)
        return (
            np.sin(curv_phases - wn_phases)
            - (
                np.exp(-np.imag(comp_wns) * length)
                * np.sin(np.real(comp_wns) * length + curv_phases - wn_phases)
            )
        ) / wn_moduli

    return (
        curv_moduli[idx1]
        * curv_moduli[idx2]
        @ (_f(comp_wns[1], curv_phases[1]) - _f(comp_wns[0], curv_phases[0]))
    )


def _egy_par(floe_params, wave_params):
    """Energy from the forcing term of the displacement ODE"""
    return _egy_par_pow2(floe_params, wave_params) + _egy_par_m(
        floe_params, wave_params
    )


def _egy_m(floe_params, wave_params):
    red_num, length = floe_params
    _, c_wavenumbers = wave_params
    adim = red_num * length
    wavenumbers = np.real(c_wavenumbers)
    attenuations = np.imag(c_wavenumbers)

    wn_abs, wn_phases = (_f(c_wavenumbers) for _f in (np.abs, np.angle))
    _, comp_curvs = _egy_par_vals(red_num, wave_params)
    curv_moduli, curv_phases = np.abs(comp_curvs), np.angle(comp_curvs)
    num_add = red_num + wavenumbers
    num_sub = red_num - wavenumbers
    att_add = red_num + attenuations
    att_sub = red_num - attenuations
    q_pp = att_add**2 + num_add**2
    q_mp = att_sub**2 + num_add**2
    q_pm = att_add**2 + num_sub**2
    q_mm = att_sub**2 + num_sub**2

    phase_deltas = wn_phases - curv_phases
    sin_phase_deltas = np.sin(phase_deltas)
    cos_phase_deltas = np.cos(phase_deltas)
    phase_quads = curv_phases + PI_D4
    sin_phase_quads = np.sin(phase_quads)
    cos_phase_quads = np.cos(phase_quads)
    energ_1_K = np.array(
        (
            sin_phase_deltas * (1 / q_mp - 1 / q_mm),
            cos_phase_deltas * (1 / q_mp + 1 / q_mm),
            -sin_phase_deltas * (1 / q_pp - 1 / q_pm),
            -cos_phase_deltas * (1 / q_pp + 1 / q_pm),
        )
    )
    energ_1_b = np.array(
        (
            -(sin_phase_quads / q_mp - cos_phase_quads / q_mm),
            (cos_phase_quads / q_mp - sin_phase_quads / q_mm),
            -(cos_phase_quads / q_pp - sin_phase_quads / q_pm),
            -(sin_phase_quads / q_pp - cos_phase_quads / q_pm),
        )
    )
    energ_1 = wn_abs * energ_1_K + SQR2 * red_num * energ_1_b
    energ_1[:2, :] *= np.exp(-adim)

    arg_add = num_add * length
    arg_add_phase = arg_add - phase_deltas
    sin_arg_ap = np.sin(arg_add_phase)
    cos_arg_ap = np.cos(arg_add_phase)
    arg_sub = num_sub * length
    arg_sub_phase = arg_sub + phase_deltas
    sin_arg_sp = np.sin(arg_sub_phase)
    cos_arg_sp = np.cos(arg_sub_phase)
    arg_add_quad = arg_add + phase_quads
    sin_arg_aq = np.sin(arg_add_quad)
    cos_arg_aq = np.cos(arg_add_quad)
    arg_sub_quad = arg_sub - curv_phases + PI_D4
    sin_arg_sq = np.sin(arg_sub_quad)
    cos_arg_sq = np.cos(arg_sub_quad)
    energ_exp_K = np.array(
        (
            sin_arg_ap / q_mp + sin_arg_sp / q_mm,
            -(cos_arg_ap / q_mp + cos_arg_sp / q_mm),
            -(sin_arg_ap / q_pp + sin_arg_sp / q_pm),
            cos_arg_ap / q_pp + cos_arg_sp / q_pm,
        )
    )
    energ_exp_b = np.array(
        (
            sin_arg_aq / q_mp - sin_arg_sq / q_mm,
            -cos_arg_aq / q_mp + cos_arg_sq / q_mm,
            cos_arg_aq / q_pp - cos_arg_sq / q_pm,
            sin_arg_aq / q_pp - sin_arg_sq / q_pm,
        )
    )
    energ_exp = wn_abs * energ_exp_K + SQR2 * red_num * energ_exp_b
    energ_exp[2:, :] *= np.exp(-adim)
    energ_exp *= np.exp(-attenuations * length)

    energ = (
        (energ_1 + energ_exp) @ curv_moduli @ _dis_hom_coefs(floe_params, wave_params)
    )
    return red_num**2 * energ


def energy(
    floe_params: tuple[float],
    wave_params: tuple[np.ndarray],
    growth_params: tuple | None = None,
    an_sol: bool | None = None,
    num_params: dict | None = None,
) -> float:
    if numerical._use_an_sol(an_sol, floe_params[1], growth_params):
        return (
            _egy_hom(floe_params, wave_params)
            + 2 * _egy_m(floe_params, wave_params)
            + _egy_par(floe_params, wave_params)
        )
    return numerical.energy(floe_params, wave_params, growth_params, num_params)[0]
