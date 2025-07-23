from __future__ import annotations

import functools
import typing

import attrs
import numpy as np

from . import numerical
from ._ph_utils import _wavefield
from .constants import PI_D4, SQR2

if typing.TYPE_CHECKING:
    from ..model import model


def _package_wuf(
    wuf: model.WavesUnderFloe, growth_params: tuple | None
) -> tuple[tuple, tuple, tuple | None]:
    floe_params = wuf.wui.ice._red_elastic_number, wuf.length
    wave_params = wuf.edge_amplitudes, wuf.wui._c_wavenumbers
    if growth_params is not None:
        growth_params = growth_params[0] - wuf.left_edge, growth_params[1]
    return floe_params, wave_params, growth_params


def _dis_par_amps(red_num: float, wave_params: tuple[np.ndarray, np.ndarray]):
    """Complex amplitudes of individual particular solutions"""
    c_amplitudes, c_wavenumbers = wave_params
    return c_amplitudes / (1 + 0.25 * (c_wavenumbers / red_num) ** 4)


def _dis_hom_coefs(
    floe_params: tuple[float, float], wave_params: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Coefficients of the four orthogonal homogeneous solutions"""
    return _dis_hom_mat(*floe_params) @ _dis_hom_rhs(floe_params, wave_params)


def _dis_hom_mat(red_num: float, length: float):
    """Linear application to determine, from the BCs, the coefficients of
    the four independent solutions to the homo ODE"""
    adim = red_num * length
    adim2 = 2 * adim
    denom = (
        -2
        * red_num**2
        * (np.expm1(-adim2) ** 2 + 2 * np.exp(-adim2) * (np.cos(adim2) - 1))
    )
    mat = np.array(
        [
            [
                SQR2 * np.exp(-adim) * np.sin(adim2 + PI_D4) - np.exp(-3 * adim),
                -SQR2 * np.cos(adim + PI_D4)
                - 3 * np.exp(-adim2) * np.sin(adim)
                + np.exp(-adim2) * np.cos(adim),
                np.exp(-adim) * np.sin(adim2) - np.exp(-adim) + np.exp(-3 * adim),
                np.cos(adim)
                - 2 * np.exp(-adim2) * np.sin(adim)
                - np.exp(-adim2) * np.cos(adim),
            ],
            [
                -SQR2 * np.exp(-adim) * np.cos(adim2 + PI_D4)
                + 2 * np.exp(-adim)
                - np.exp(-3 * adim),
                -SQR2 * np.sin(adim + PI_D4)
                + SQR2 * np.exp(-adim2) * np.cos(adim + PI_D4),
                -np.exp(-adim) * np.cos(adim2) + np.exp(-adim),
                np.sin(adim) - np.exp(-adim2) * np.sin(adim),
            ],
            [
                -1 + SQR2 * np.exp(-adim2) * np.cos(adim2 + PI_D4),
                (3 * np.sin(adim) + np.cos(adim)) * np.exp(-adim)
                - SQR2 * np.exp(-3 * adim) * np.sin(adim + PI_D4),
                (np.sin(adim2) + 1) * np.exp(-adim2) - 1,
                (-2 * np.sin(adim) + np.cos(adim)) * np.exp(-adim)
                - np.exp(-3 * adim) * np.cos(adim),
            ],
            [
                (SQR2 * np.sin(adim2 + PI_D4) - 2) * np.exp(-adim2) + 1,
                -SQR2 * np.exp(-adim) * np.sin(adim + PI_D4)
                + SQR2 * np.exp(-3 * adim) * np.cos(adim + PI_D4),
                (-np.cos(adim2) + 1) * np.exp(-adim2),
                np.exp(-adim) * np.sin(adim) - np.exp(-3 * adim) * np.sin(adim),
            ],
        ]
    )
    mat[:, 2:] /= red_num

    return mat / denom


def _dis_hom_rhs(
    floe_params: tuple[float, float], wave_params: tuple[np.ndarray, np.ndarray]
):
    """Vector onto which apply the matrix, to extract the coefficients"""
    red_num, length = floe_params
    _, c_wavenumbers = wave_params
    exp_arg = 1j * c_wavenumbers * length

    r1 = c_wavenumbers**2 * _dis_par_amps(red_num, wave_params)
    r2 = np.imag(r1 @ np.exp(exp_arg))
    r3 = 1j * c_wavenumbers * r1
    r4 = np.imag(r3 @ np.exp(exp_arg))
    r1 = np.imag(r1).sum()
    r3 = np.imag(r3).sum()

    return np.array((r1, r2, r3, r4))


# Can be used to decorate methods to make them return a scalar instead of
# a 1d-array of length 1, if the argument is a scalar OR a 0d array (that is, a
# scalar). If the argument is a tuple or a list or a 1d array of length 1,
# a 1d-array of length 1 is returned.
# This emulate the behaviour of standard numpy ufuncs.
def _demote_to_scalar(f):
    @functools.wraps(f)
    def wrapper(*args):
        instance, x = args
        x = np.asarray(x)
        res = f(instance, x)
        if len(res) == 1 and x.ndim == 0:
            return res.item()
        return res

    return wrapper


@attrs.define
class FluidSurfaceHandler:
    wave_params: tuple[np.ndarray, np.ndarray]
    growth_params: tuple[np.ndarray, float] | None = None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*(_package_wuf(wuf, growth_params)[1:]))

    @classmethod
    def from_domain(cls, domain: model.Domain, growth_params: tuple | None = None):
        return cls(
            (
                domain.spectrum.amplitudes * np.exp(1j * domain.spectrum.phases),
                domain.fsw.wavenumbers,
            ),
            growth_params,
        )

    @_demote_to_scalar
    def compute(self, x, /):
        return numerical.free_surface(x, self.wave_params, self.growth_params)


@attrs.define
class DisplacementHandler:
    floe_params: tuple[float, float]
    wave_params: tuple[np.ndarray, np.ndarray]
    growth_params: tuple[np.ndarray, float] | None = None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*_package_wuf(wuf, growth_params))

    def _dis_hom(self, x: np.ndarray):
        """Homogeneous solution to the displacement ODE"""
        red_num, length = self.floe_params
        arr = red_num * x
        cosx, sinx = np.cos(arr), np.sin(arr)
        expx = np.exp(-red_num * (length - x))
        exmx = np.exp(-arr)
        return np.vstack(
            ([expx * cosx, expx * sinx, exmx * cosx, exmx * sinx])
        ).T @ _dis_hom_coefs(self.floe_params, self.wave_params)

    def _dis_par(self, x: np.ndarray):
        """Sum of the particular solutions to the displacement ODE"""
        return _wavefield(
            x, _dis_par_amps(self.floe_params[0], self.wave_params), self.wave_params[1]
        )

    @_demote_to_scalar
    def _dis(self, x, /):
        return self._dis_hom(x) + self._dis_par(x)

    def compute(
        self,
        x,
        an_sol: bool | None = None,
        num_params: dict | None = None,
    ):
        if numerical._use_an_sol(an_sol, self.floe_params[1], self.growth_params):
            return self._dis(np.asarray(x))
        return numerical.displacement(
            x, self.floe_params, self.wave_params, self.growth_params, num_params
        )


@attrs.define
class CurvatureHandler:
    floe_params: tuple[float, float]
    wave_params: tuple[np.ndarray, np.ndarray]
    growth_params: tuple[np.ndarray, float] | None = None

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(*_package_wuf(wuf, growth_params))

    def _cur_wavefield(self, x):
        """Second derivative of the interface"""
        red_num = self.floe_params[0]
        _, c_wavenumbers = self.wave_params

        return -np.imag(
            (_dis_par_amps(red_num, self.wave_params) * c_wavenumbers**2)
            @ np.exp(1j * c_wavenumbers[:, None] * x)
        )

    def _cur_hom(self, x):
        """Second derivative of the homogeneous part of the displacement"""
        red_num, length = self.floe_params
        arr = red_num * x
        cosx, sinx = np.cos(arr), np.sin(arr)
        expx = np.exp(-red_num * (length - x))
        exmx = np.exp(-arr)
        return (
            2
            * red_num**2
            * np.vstack(([-expx * sinx, expx * cosx, exmx * sinx, -exmx * cosx])).T
            @ _dis_hom_coefs(self.floe_params, self.wave_params)
        )

    def _cur_par(self, x):
        """Second derivative of the particular part of the displacement"""
        return self._cur_wavefield(x)

    @_demote_to_scalar
    def _cur(self, x, /):
        return self._cur_hom(x) + self._cur_par(x)

    def compute(
        self,
        x,
        an_sol: bool | None = None,
        num_params: dict | None = None,
        is_linear: bool = True,
    ):
        """Curvature of the floe.

        A linear approximation is given by the second derivative of the
        displacement. The exact definition involves the first derivative of the
        displacement.

        Parameters
        ----------
        x : float or 1d array_like of float
            Along-floe coordinates at which to compute the curvature, in m
        an_sol : bool
            Whether to use an analytical formulation of the curvature
        num_params : dict
            Optional arguments passed to the numerical solver if `not an_sol`
        is_linear : bool
            Whether to use the linear approximation; ignored if `an_sol`

        Returns
        -------
        float or 1d array_like of float

        """
        if numerical._use_an_sol(
            an_sol, self.floe_params[1], self.growth_params, is_linear
        ):
            return self._cur(x)
        return numerical.curvature(
            x,
            self.floe_params,
            self.wave_params,
            self.growth_params,
            num_params,
            is_linear,
        )


@attrs.define
class StrainHandler:
    curv_handler: CurvatureHandler
    thickness: float

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        return cls(CurvatureHandler.from_wuf(wuf, growth_params), wuf.wui.ice.thickness)

    def compute(self, x, an_sol: bool | None = None, num_params: dict | None = None):
        return -self.thickness / 2 * self.curv_handler.compute(x, an_sol, num_params)


@attrs.define
class EnergyHandler:
    floe_params: tuple[float, float]
    wave_params: tuple[np.ndarray, np.ndarray]
    growth_params: tuple[np.ndarray, float] | None = None
    factor: float = 1

    @classmethod
    def from_wuf(cls, wuf: model.WavesUnderFloe, growth_params=None):
        factor = cls._compute_factor(wuf)
        return cls(*_package_wuf(wuf, growth_params), factor)

    @staticmethod
    def _compute_factor(wuf: model.WavesUnderFloe):
        # NOTE: this could also be a WUF property?
        return wuf.wui.ice.flex_rigidity / (2 * wuf.wui.ice.thickness)

    def _egy_hom(self):
        """Energy from the homogen term of the displacement ODE"""
        red_num, length = self.floe_params
        adim = red_num * length
        adim2 = 2 * adim
        c_1, c_2, c_3, c_4 = _dis_hom_coefs(self.floe_params, self.wave_params)

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

    def _egy_par_vals(self):
        red_num = self.floe_params[0]
        comp_amps = _dis_par_amps(red_num, self.wave_params)
        _, c_wavenumbers = self.wave_params

        return comp_amps * (1j * c_wavenumbers) ** 2

    def _egy_par_pow2(self):
        """Energy contribution from individual forcings"""
        red_num, length = self.floe_params
        _, c_wavenumbers = self.wave_params
        wavenumbers = np.real(c_wavenumbers)
        attenuations = np.imag(c_wavenumbers)

        comp_curvs = self._egy_par_vals()
        wn_moduli, curv_moduli = map(np.abs, (c_wavenumbers, comp_curvs))
        wn_phases, curv_phases = map(np.angle, (c_wavenumbers, comp_curvs))

        # Because of a division by `attenuations`, this term must me handled
        # with some extra care
        attenuation_mask = attenuations != 0
        non_oscillatory_term = np.full(attenuations.shape, np.nan)
        non_oscillatory_term[attenuation_mask] = (
            -np.expm1(-2 * attenuations[attenuation_mask] * length)
            / attenuations[attenuation_mask]
        )
        # NOTE: corresponds to the limit of the previous term, when attenuation -> 0
        non_oscillatory_term[~attenuation_mask] = 2 * length

        return (
            curv_moduli**2
            @ (
                non_oscillatory_term
                + (
                    np.sin(2 * curv_phases - wn_phases)
                    - np.sin(2 * (wavenumbers * length + curv_phases) - wn_phases)
                    * np.exp(-2 * attenuations * length)
                )
                / wn_moduli
            )
        ) / 4

    def _egy_par_m(self):
        """Energy contribution from forcing interactions"""
        red_num, length = self.floe_params
        _, c_wavenumbers = self.wave_params
        wavenumbers = np.real(c_wavenumbers)
        attenuations = np.imag(c_wavenumbers)
        comp_curvs = self._egy_par_vals()

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

    def _egy_par(self):
        """Energy from the forcing term of the displacement ODE"""
        return self._egy_par_pow2() + self._egy_par_m()

    def _egy_m(self):
        red_num, length = self.floe_params
        _, c_wavenumbers = self.wave_params
        adim = red_num * length
        wavenumbers = np.real(c_wavenumbers)
        attenuations = np.imag(c_wavenumbers)

        wn_abs, wn_phases = (_f(c_wavenumbers) for _f in (np.abs, np.angle))
        comp_curvs = self._egy_par_vals()
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
            (energ_1 + energ_exp)
            @ curv_moduli
            @ _dis_hom_coefs(self.floe_params, self.wave_params)
        )
        return red_num**2 * energ

    def compute(
        self,
        an_sol: bool | None = None,
        num_params: dict | None = None,
        integration_method: str | None = None,
        linear_curvature: bool | None = None,
    ) -> float:
        do_use_an_sol = numerical._use_an_sol(
            an_sol, self.floe_params[1], self.growth_params, linear_curvature
        )
        if do_use_an_sol:
            unit_energy = self._egy_hom() + 2 * self._egy_m() + self._egy_par()
        else:
            linear_curvature = True if linear_curvature else False
            # TODO: documenter le comportement par défault.
            # Inclure un comportement différent en mono (systématique) et poly
            # (stochastic), avec un warning si l'utilisation d'une méthode
            # systématique est utilisée en poly, car ça pourrait prendre un
            # temps infini.
            unit_energy = numerical.unit_energy(
                self.floe_params,
                self.wave_params,
                self.growth_params,
                num_params,
                integration_method,
                linear_curvature,
            )

        return self.factor * unit_energy
