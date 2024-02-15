#!/usr/bin/env python3

from __future__ import annotations

import functools
import itertools
import warnings
import numpy as np
import scipy.optimize as optimize

# from .libraries.WaveUtils import SpecVars
from .pars import g


PI = np.pi


class Wave:
    """Represents a monochromatic wave."""

    def __init__(
        self,
        amplitude: float,
        *,
        period: float = None,
        frequency: float = None,
        phase: float = 0,
        beta: float = 0,
    ):
        """Initialise self."""
        if period is None and frequency is None:
            raise ValueError("Either period or frequency must be specified.")
        elif period is not None:
            self.__period = period
            if frequency is not None:
                warnings.warn(
                    (
                        "Both period and frequency were specified, "
                        "frequency will be ignored"
                    ),
                    stacklevel=2,
                )
            self.__frequency = 1 / period
        else:
            self.__frequency = frequency
            self.__period = 1 / frequency

        self.__amplitude = amplitude
        self.__phase = phase
        self.__beta = beta

    @property
    def amplitude(self) -> float:
        """Wave amplitude in m."""
        return self.__amplitude

    @property
    def period(self) -> float:
        """Wave period in s."""
        return self.__period

    @property
    def phase(self) -> float:
        return self.__phase

    @property
    def frequency(self) -> float:
        """Wave frequency in Hz."""
        return self.__frequency

    @functools.cached_property
    def angular_frequency(self) -> float:
        """Wave angular frequency in rad s**-1."""
        return 2 * PI * self.frequency

    @functools.cached_property
    def angular_frequency2(self) -> float:
        """Squared wave angular frequency, for convenience."""
        return self.angular_frequency**2


class Ocean:
    """Represents the fluid bearing the floes.

    Encapsulates the properties of an incompressible ocean of finite,
    constant depth and given density.

    Parameters
    ----------
    depth : float
        Ocean constant and finite depth in m.
    density : float
        Ocean constant density in kg m**-3.

    Attributes
    ----------
    density : float
    depth : float

    """

    def __init__(self, depth: float = 1000, density: float = 1025) -> None:
        """Initialise self."""
        self.__depth = depth
        self.__density = density

    @property
    def density(self) -> float:
        """Ocean density in kg m**-3."""
        return self.__density

    @property
    def depth(self) -> float:
        """Ocean depth in m."""
        return self.__depth


class Ice:
    def __init__(
        self,
        density: float = 922.5,
        frac_energy: float = 1e5,
        poissons_ratio: float = 0.3,
        thickness: float = 1.0,
        youngs_modulus: float = 6e9,
    ):
        self.__density = density
        self.__frac_energy = frac_energy
        self.__poissons_ratio = poissons_ratio
        self.__thickness = thickness
        self.__youngs_modulus = youngs_modulus

    @property
    def density(self):
        return self.__density

    @property
    def frac_energy(self):
        return self.__frac_energy

    @property
    def poissons_ratio(self):
        return self.__poissons_ratio

    @property
    def thickness(self):
        return self.__thickness

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus

    @functools.cached_property
    def quad_moment(self):
        return self.thickness**3 / (12 * (1 - self.poissons_ratio**2))

    @functools.cached_property
    def flex_rigidity(self):
        return self.quad_moment * self.youngs_modulus

    @functools.cached_property
    def frac_toughness(self):
        return (1 - self.poissons_ratio**2) * self.frac_energy**2 / self.youngs_modulus


class IceCoupled(Ice):
    def __init__(
        self,
        ice: Ice,
        ocean: OceanCoupled,
        spec: DiscreteSpectrum,
        dispersion: str,
        gravity: float,
    ):
        super().__init__(
            ice.density,
            ice.frac_energy,
            ice.poissons_ratio,
            ice.thickness,
            ice.youngs_modulus,
        )
        self.__draft = self.thickness * self.density / ocean.density
        self.__dud = ocean.depth - self.draft
        self._elastic_length_pow4 = self.flex_rigidity / (ocean.density * gravity)
        self.__elastic_length = self._elastic_length_pow4 ** (1 / 4)
        self.__wavenumbers = self.compute_wavenumbers(ocean, spec, gravity)

    @property
    def draft(self):
        return self.__draft

    @property
    def dud(self):
        return self.__dud

    @property
    def elastic_length(self):
        return self.__elastic_length

    @property
    def wavenumbers(self):
        return self.__wavenumbers

    @functools.cached_property
    def freeboard(self):
        return self.thickness - self.draft

    @functools.cached_property
    def attenuations(self):
        return self.wavenumbers**2 * self.thickness / 4

    @functools.cached_property
    def _red_elastic_number(self):
        return 1 / (2**0.5 * self.elastic_length)

    def compute_wavenumbers(
        self, ocean: OceanCoupled, ds: DiscreteSpectrum, gravity: float
    ) -> np.ndarray:
        return self._comp_wns(ocean.density, ds._ang_freq2, gravity)

    def _comp_wns(
        self, density: float, angfreqs2: np.ndarray, gravity: float
    ) -> np.ndarray:
        def f(kk: float, d0: float, d1: float, rr: float) -> float:
            obj = (kk**5 + d1 * kk) * np.tanh(rr * kk) + d0
            return obj

        def df_dk(kk: float, d0: float, d1: float, rr: float) -> float:
            return (5 * kk**4 + d1 + rr * d0) * np.tanh(rr * kk) + rr * (
                kk**5 + d1 * kk
            )

        def extract_real_root(roots):
            mask = (np.imag(roots) == 0) & (np.real(roots) > 0)
            if mask.nonzero()[0].size != 1:
                raise ValueError("An approximate initial guess could not be found")
            return np.real(roots[mask][0])

        def find_k(k0, alpha, d0, d1, rr):
            res = optimize.root_scalar(
                f,
                args=(d0, d1, rr),
                fprime=df_dk,
                x0=k0,
            )
            if not res.converged:
                warnings.warn(
                    f"Root finding did not converge: ice-covered surface, "
                    f"f={np.sqrt(alpha*gravity)/(2*PI):1.2g} Hz",
                    stacklevel=2,
                )
            return res.root

        scaled_ratio = self.dud / self.elastic_length

        alphas = angfreqs2 / gravity
        deg1 = 1 - alphas * self.draft
        deg0 = -alphas * self.elastic_length
        roots = np.full(angfreqs2.size, np.nan)

        for i, (alpha, _d0, _d1) in enumerate(zip(alphas, deg0, deg1)):
            find_k_i = functools.partial(
                find_k,
                alpha=alpha,
                d0=_d0,
                d1=_d1,
                rr=scaled_ratio,
            )

            # We always expect one positive real root,
            # and if _d1 < 0, eventually two additional negative real roots.
            roots_dw = np.polynomial.polynomial.polyroots([_d0, _d1, 0, 0, 0, 1])
            k0_dw = extract_real_root(roots_dw)
            if np.isposinf(self.dud):
                roots[i] = k0_dw
                continue
            if scaled_ratio * k0_dw > 1.47:  # tanh(1.47) approx 0.9
                roots[i] = find_k_i(k0_dw)
            else:
                roots_sw = np.polynomial.polynomial.polyroots(
                    [_d0 / scaled_ratio, 0, _d1, 0, 0, 0, 1]
                )
                k0_sw = extract_real_root(roots_sw)

                if scaled_ratio * k0_sw < 0.11:
                    roots[i] = find_k_i(k0_sw)
                else:
                    roots[i] = find_k_i((k0_dw + k0_sw) / 2)

        return roots / self.elastic_length


class Floe:
    def __init__(
        self,
        left_edge: float,
        length: float,
        ice: Ice,
        dispersion: str = "ElML",
    ):
        self.__left_edge = left_edge
        self.__length = length
        self.__ice = ice

    @property
    def left_edge(self):
        return self.__left_edge

    @property
    def length(self):
        return self.__length

    @property
    def ice(self):
        return self.__ice

    @property
    def dispersion(self):
        return self.__dispersion


class FloeCoupled(Floe):
    def __init__(
        self,
        floe: Floe,
        ice: IceCoupled,
        spectrum: DiscreteSpectrum,
        phases: np.ndarray | list[float] | float,
        dispersion=None,
    ):
        super().__init__(floe.left_edge, floe.length, ice, dispersion)
        self.__phases = np.asarray(phases)

    @property
    def phases(self) -> np.ndarray:
        return self.__phases

    # TODO
    # @Floe.ice.getter
    # def ice(self) -> IceCoupled:
    #     return self.__ice

    @functools.cached_property
    def _adim(self):
        return self.length * self.ice._red_elastic_number

    def _dis_par_amps(self, amplitudes: np.ndarray):
        """Complex amplitudes of individual particular solutions"""
        return -(
            np.exp(1j * self.phases)
            * amplitudes
            / (
                1
                + (
                    self.ice._elastic_length_pow4
                    * (self.ice.attenuations - 1j * self.ice.wavenumbers) ** 4
                )
            )
        )

    def _dis_hom_mat(self):
        """Linear application to determine, from the BCs, the coefficients of
        the four independent solutions to the homo ODE"""
        red_el_num = self.ice._red_elastic_number
        adim_floe = red_el_num * self.length

        reduc_denom = (
            1
            + 2 * np.exp(-2 * adim_floe) * (np.cos(2 * adim_floe) - 2)
            + np.exp(-4 * adim_floe)
        )

        m1 = (
            (
                1
                - 2 * np.exp(-2 * adim_floe) * np.cos(2 * adim_floe)
                + np.exp(-4 * adim_floe)
            )
            / reduc_denom
            - 1j
        ) / (4 * red_el_num**2)
        m2 = (
            np.expm1(-2 * adim_floe)
            * np.exp(-adim_floe)
            * np.sin(adim_floe)
            / red_el_num**2
            / reduc_denom
        )
        m3 = (
            (
                1
                - 2 * np.exp(-2 * adim_floe) * np.sin(2 * adim_floe)
                - np.exp(-4 * adim_floe)
            )
            / reduc_denom
            / (4 * red_el_num**3)
        )
        m4 = (
            (
                np.exp(-adim_floe)
                * (
                    np.sin(adim_floe)
                    - np.cos(adim_floe)
                    + np.exp(-2 * adim_floe) * (np.sin(adim_floe) + np.cos(adim_floe))
                )
            )
            / reduc_denom
            / (2 * red_el_num**3)
        )

        m5 = (
            (-1 + 1j)
            * (
                1
                + 2 * np.exp(-2 * adim_floe) * np.sin(2 * adim_floe)
                - np.exp(-4 * adim_floe)
            )
            / reduc_denom
            / (4 * red_el_num**2)
        )
        m6 = (
            (1 - 1j)
            * (
                np.exp(-adim_floe)
                * (
                    np.sin(adim_floe)
                    + np.cos(adim_floe)
                    + np.exp(-2 * adim_floe) * (np.sin(adim_floe) - np.cos(adim_floe))
                )
            )
            / reduc_denom
            / (2 * red_el_num**2)
        )
        m7 = (
            -(
                np.expm1(-2 * adim_floe) ** 2
                + 2j * np.exp(-2 * adim_floe) * (np.cos(2 * adim_floe) - 1)
            )
            / reduc_denom
            / (4 * red_el_num**3)
        )
        m8 = (
            (1 - 1j)
            * np.expm1(-2 * adim_floe)
            * np.exp(-adim_floe)
            * np.sin(adim_floe)
            / (2 * red_el_num**3)
            / reduc_denom
        )

        mat = np.full((4, 4), np.nan, dtype=complex)
        mat[0] = m1, m2, m3, m4
        mat[2] = m5, m6, m7, m8
        mat[1::2] = np.conj(mat[0::2])
        return mat

    def _dis_hom_rhs(self, amplitudes):
        """Vector onto which apply the matrix, to extract the coefficients"""
        exp_mod = -self.ice.attenuations + 1j * self.ice.wavenumbers

        r1 = self._dis_par_amps(amplitudes) * exp_mod**2
        r2 = np.imag(r1 @ np.exp(exp_mod * self.length))
        r3 = r1 * exp_mod
        r4 = np.imag(r3 @ np.exp(exp_mod * self.length))
        r1 = np.imag(r1).sum()
        r3 = np.imag(r3).sum()

        return -np.array((r1, r2, r3, r4))

    def _dis_hom_coefs(self, amplitudes: np.ndarray) -> np.ndarray:
        """Coefficients of the four orthogonal homogeneous solutions"""
        return self._dis_hom_mat() @ self._dis_hom_rhs(amplitudes)

    def surface(self, x, spectrum):
        return (
            self._wavefield(x, spectrum._amps * np.exp(1j * self.phases))
            - self.ice.draft
        )

    def _wavefield(self, x, complex_amps):
        """Vertical coordinate of the floe--wave interface

        `complex_amps` is left free, so it can be used with natural spectral amplitudes,
        or user-provided ones, incorporating a phase

        """
        # TODO: could be better of in DiscreteSpectrum
        return np.imag(
            complex_amps
            @ np.exp((-self.ice.attenuations + 1j * self.ice.wavenumbers)[:, None] * x)
        )

    def _mean_wavefield(self, amplitudes):
        """Mean interface of the floe-attenuated wave"""
        comp_wn = -self.ice.attenuations + 1j * self.ice.wavenumbers
        return (
            np.imag(
                amplitudes
                * np.exp(1j * self.phases)
                / comp_wn
                * (np.exp(comp_wn * self.length) - 1)
            ).sum()
            / self.length
        )

    def _dis_hom(self, x, amplitudes: np.ndarray):
        """Homogeneous solution to the displacement ODE"""
        red_el_num = self.ice._red_elastic_number
        return np.real(
            self._dis_hom_coefs(amplitudes)
            @ (
                np.cosh((1 + 1j) * red_el_num * x),
                np.cosh((1 - 1j) * red_el_num * x),
                np.sinh((1 + 1j) * red_el_num * x),
                np.sinh((1 - 1j) * red_el_num * x),
            )
        )

    def _dis_par(self, x, amplitudes):
        """Sum of the particular solutions to the displacement ODE"""
        return self._mean_wavefield(amplitudes) + self._wavefield(
            x, self._dis_par_amps(amplitudes)
        )

    def _displacement(self, x, amplitudes):
        return self._dis_hom(x, amplitudes) + self._dis_par(x, amplitudes)

    def displacement(self, x, spectrum):
        """Complete solution of the displacement ODE

        `x` is expected to be relative to the floe, i.e. to be bounded by 0, L
        """
        return self._displacement(x, spectrum._amps)

    def _cur_wavefield(self, x, spectrum, complex_amps):
        """Second derivative of the interface"""
        return np.imag(
            (complex_amps * (-self.ice.attenuations + 1j * self.ice.wavenumbers) ** 2)
            @ np.exp((-self.ice.attenuations + 1j * self.ice.wavenumbers)[:, None] * x)
        )

    def _cur_hom(self, x, spectrum):
        """Second derivative of the homogeneous part of the displacement"""
        red_el_num = self.ice._red_elastic_number
        return np.real(
            self._dis_hom_coefs(spectrum)
            @ (
                ((1 + 1j) * red_el_num) ** 2 * np.cosh((1 + 1j) * red_el_num * x),
                ((1 - 1j) * red_el_num) ** 2 * np.cosh((1 - 1j) * red_el_num * x),
                ((1 + 1j) * red_el_num) ** 2 * np.sinh((1 + 1j) * red_el_num * x),
                ((1 - 1j) * red_el_num) ** 2 * np.sinh((1 - 1j) * red_el_num * x),
            )
        )

    def _cur_par(self, x, spectrum):
        """Second derivative of the particular part of the displacement"""
        return self._cur_wavefield(x, spectrum, self._dis_par_amps(spectrum))

    def curvature(self, x, spectrum):
        """Curvature of the floe, i.e. second derivative of the vertical displacement"""
        return self._cur_hom(x, spectrum) + self._cur_par(x, spectrum)

    def _egy_hom_c(self, spectrum, x0):
        """Energy contribution from the cosh eigenfunction"""
        self.ice._red_elastic_number
        adim2 = 2 * self.length * self.ice._red_elastic_number

        _c = self._dis_hom_coefs(spectrum)[0]
        real, imag = np.real(_c), np.imag(_c)

        return (
            imag**2
            * (
                4 * self.length
                + (
                    np.sinh(adim2) * (2 + np.cos(adim2))
                    + np.sin(adim2) * (2 + np.cosh(adim2))
                )
                / self.ice._red_elastic_number
            )
            + 2
            * real
            * imag
            * (np.cosh(adim2) * np.sin(adim2) - np.sinh(adim2) * np.cos(adim2))
            / self.ice._red_elastic_number
            - real**2
            * (
                4 * self.length
                - (
                    np.sinh(adim2) * (2 - np.cos(adim2))
                    + np.sin(adim2) * (2 - np.cosh(adim2))
                )
                / self.ice._red_elastic_number
            )
        )

    def _egy_hom_s(self, spectrum):
        """Energy contribution from the sinh eigenfunction"""
        self.ice._red_elastic_number
        adim2 = 2 * self.length * self.ice._red_elastic_number

        _c = self._dis_hom_coefs(spectrum)[2]
        real, imag = np.real(_c), np.imag(_c)

        return (
            imag**2
            * (
                -4 * self.length
                + (
                    np.sinh(adim2) * (2 + np.cos(adim2))
                    - np.sin(adim2) * (2 - np.cosh(adim2))
                )
                / self.ice._red_elastic_number
            )
            + 2
            * real
            * imag
            * (np.cosh(adim2) * np.sin(adim2) - np.sinh(adim2) * np.cos(adim2))
            / self.ice._red_elastic_number
            + real**2
            * (
                4 * self.length
                + (
                    np.sinh(adim2) * (2 - np.cos(adim2))
                    - np.sin(adim2) * (2 + np.cosh(adim2))
                )
                / self.ice._red_elastic_number
            )
        )

    def _egy_hom_m(self, spectrum):
        """Energy contribution from eigenfunctions interaction"""
        adim2 = 2 * self._adim

        c1, c2 = self._dis_hom_coefs(spectrum)[[0, 2]]
        x1, x2 = map(np.real, (c1, c2))
        y1, y2 = map(np.imag, (c1, c2))

        return (
            y1
            * y2
            * (
                np.cosh(adim2) * (2 + np.cos(adim2))
                + np.sinh(adim2) * np.sin(adim2)
                - 3
            )
            - y1
            * x2
            * (
                np.cos(adim2) * (2 + np.cosh(adim2))
                - np.sinh(adim2) * np.sin(adim2)
                - 3
            )
            + x1
            * y2
            * (
                np.cos(adim2) * (2 - np.cosh(adim2))
                + np.sinh(adim2) * np.sin(adim2)
                - 1
            )
            + x1
            * x2
            * (
                np.cosh(adim2) * (2 - np.cos(adim2))
                - np.sinh(adim2) * np.sin(adim2)
                - 1
            )
        ) / self.ice._red_elastic_number

    def _egy_hom(self, spectrum):
        """Energy from the homogen term of the displacement ODE"""
        return (
            self._egy_hom_c(spectrum)
            + 2 * self._egy_hom_m(spectrum)
            + self._egy_hom_s(spectrum)
        ) / (4 * self.ice._elastic_length_pow4)

    def _egy_par_vals(self, spectrum):
        comp_amps = self._dis_par_amps(spectrum)
        comp_wns = self.wavenumbers + 1j * self.ice.attenuations

        comp_curvs = comp_amps * (1j * comp_wns) ** 2

        return comp_wns, comp_curvs

    def _egy_par_pow2(self, spectrum):
        """Energy contribution from individual forcings"""
        comp_wns, comp_curvs = self._egy_par_vals(spectrum)
        wn_moduli, curv_moduli = map(np.abs, (comp_wns, comp_curvs))
        wn_phases, curv_phases = map(np.angle, (comp_wns, comp_curvs))

        red = np.exp(-2 * self.ice.attenuations * self.length)

        return (
            curv_moduli**2
            @ (
                (1 - red) / self.ice.attenuations
                + (
                    np.sin(2 * curv_phases - wn_phases)
                    - np.sin(
                        2 * (self.wavenumbers * self.length + curv_phases) - wn_phases
                    )
                    * red
                )
                / wn_moduli
            )
        ) / 4

    def _egy_par_m(self, spectrum):
        """Energy contribution from forcing interactions"""
        _, comp_curvs = self._egy_par_vals(spectrum)

        # Binomial coefficients, much more efficient than itertools
        idx1, idx2 = np.triu_indices(
            spectrum.nf, 1
        )  # TODO: where does nf come from, amps.size?

        mean_attenuations = self.ice.attenuations[idx1] + self.ice.attenuations[idx2]
        comp_wns = (
            self.wavenumbers[idx1] - self.wavenumbers[idx2],
            self.wavenumbers[idx1] + self.wavenumbers[idx2],
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
                    np.exp(-np.imag(comp_wns) * self.length)
                    * np.sin(np.real(comp_wns) * self.length + curv_phases - wn_phases)
                )
            ) / wn_moduli

        return (
            curv_moduli[idx1]
            * curv_moduli[idx2]
            @ (_f(comp_wns[1], curv_phases[1]) - _f(comp_wns[0], curv_phases[0]))
        )

    def _egy_par(self, spectrum):
        """Energy from the forcing term of the displacement ODE"""
        return self._egy_par_pow2(spectrum) + self._egy_par_m(spectrum)

    def _egy_m(self, spectrum):
        def int_cosh_cos():
            return curv_moduli @ (
                (
                    np.cos(curv_phases)
                    * wns
                    * (kcm2**3 - st_2_sq2 * (3 * self.ice.attenuations**2 - wns**2))
                    + np.sin(curv_phases)
                    * self.ice.attenuations
                    * (kcm2**3 + st_2_sq2 * (self.ice.attenuations**2 - 3 * wns**2))
                )
                / q_base
                - (
                    np.cos(prop_phases)
                    * np.cos(adim)
                    * wns
                    * (
                        (kcm2 + 2 * self.ice.attenuations * self.ice._red_elas_number)
                        * edp
                        / qdec_p
                        + (kcm2 - 2 * self.ice.attenuations * self.ice._red_elas_number)
                        * edm
                        / qdec_m
                    )
                    / 2
                )
                + (
                    np.sin(prop_phases)
                    * np.sin(adim)
                    * self.ice._red_elas_number
                    * (bigkmp * edp / qdec_p + bigkmm * edm / qdec_m)
                    / 2
                )
                - (
                    np.sin(prop_phases)
                    * np.cos(adim)
                    * (bigkpp * decp * edp / qdec_p + bigkpm * decm * edm / qdec_m)
                    / 2
                )
                + (
                    np.cos(prop_phases)
                    * np.sin(adim)
                    * stk
                    * (decp * edp / qdec_p + decm * edm / qdec_m)
                )
            )

        def int_cosh_sin():
            return curv_moduli @ (
                (
                    np.cos(curv_phases)
                    * 2
                    * self.ice.attenuations
                    * stk
                    * (kcm2**2 + 2 * st_2_sq * kcm2_diff - st_2_sq2)
                    + np.sin(curv_phases)
                    * (
                        self.ice._red_elas_number
                        * (
                            st_2_sq**3
                            + st_2_sq2 * kcm2_diff
                            + st_2_sq
                            * (kcm2_diff**2 - (2 * self.ice.attenuations * wns) ** 2)
                            + kcm2**2 * kcm2_diff
                        )
                    )
                )
                / q_base
                - np.cos(prop_phases)
                * np.cos(adim)
                * (stk * ((decp * edp / qdec_p) + (decm * edm / qdec_m)))
                - np.sin(prop_phases)
                * np.sin(adim)
                * (
                    ((decp * bigkpp * edp / qdec_p) + (decm * bigkpm * edm / qdec_m))
                    / 2
                )
                - np.sin(prop_phases)
                * np.cos(adim)
                * (
                    self.ice._red_elas_number
                    * ((bigkmp * edp / qdec_p) + (bigkmm * edm / qdec_m))
                    / 2
                )
                - np.cos(prop_phases)
                * np.sin(adim)
                * (
                    wns
                    * (
                        (
                            (
                                kcm2
                                + 2 * self.ice.attenuations * self.ice._red_elas_number
                            )
                            * edp
                            / qdec_p
                        )
                        + (
                            (
                                kcm2
                                - 2 * self.ice.attenuations * self.ice._red_elas_number
                            )
                            * edm
                            / qdec_m
                        )
                    )
                    / 2
                )
            )

        def int_sinh_cos():
            return curv_moduli @ (
                (
                    np.cos(curv_phases)
                    * 2
                    * self.ice.attenuations
                    * stk
                    * (kcm2**2 - 2 * st_2_sq * (st_2_sq / 2 + kcm2_diff))
                    - np.sin(curv_phases)
                    * self.ice._red_elas_number
                    * (
                        st_2_sq**3
                        - st_2_sq2 * kcm2_diff
                        + st_2_sq
                        * (kcm2_diff**2 - (2 * self.ice.attenuations * wns) ** 2)
                        - kcm2**2 * kcm2_diff
                    )
                )
                / q_base
                + np.cos(prop_phases)
                * np.cos(adim)
                * wns
                * (
                    (kcm2 + 2 * self.ice.attenuations * self.ice._red_elas_number)
                    * edp
                    / qdec_p
                    - (kcm2 - 2 * self.ice.attenuations * self.ice._red_elas_number)
                    * edm
                    / qdec_m
                )
                / 2
                - np.sin(prop_phases)
                * np.sin(adim)
                * self.ice._red_elas_number
                * (bigkmp * edp / qdec_p - bigkmm * edm / qdec_m)
                / 2
                + np.sin(prop_phases)
                * np.cos(adim)
                * (decp * bigkpp * edp / qdec_p - decm * bigkpm * edm / qdec_m)
                / 2
                - np.cos(prop_phases)
                * np.sin(adim)
                * stk
                * (decp * edp / qdec_p - decm * edm / qdec_m)
            )

        def int_sinh_sin():
            return curv_moduli @ (
                np.cos(curv_phases)
                * st_2_sq
                * wns
                * (kcm2 * (3 * self.ice.attenuations**2 - wns**2) - st_2_sq2)
                / q_base
                + np.sin(curv_phases)
                * st_2_sq
                * self.ice.attenuations
                * (kcm2 * (self.ice.attenuations**2 - 3 * wns**2) + st_2_sq2)
                / q_base
                + np.cos(prop_phases)
                * np.cos(adim)
                * stk
                * (decp * edp / qdec_p - decm * edm / qdec_m)
                + np.sin(prop_phases)
                * np.sin(adim)
                * (decp * bigkpp * edp / qdec_p - decm * bigkpm * edm / qdec_m)
                / 2
                + np.sin(prop_phases)
                * np.cos(adim)
                * self.ice._red_elas_number
                * (bigkmp * edp / qdec_p - bigkmm * edm / qdec_m)
                / 2
                + np.cos(prop_phases)
                * np.sin(adim)
                * wns
                * (
                    (kcm2 + 2 * self.ice.attenuations * self.ice._red_elas_number)
                    * edp
                    / qdec_p
                    - (kcm2 - 2 * self.ice.attenuations * self.ice._red_elas_number)
                    * edm
                    / qdec_m
                )
                / 2
            )

        adim = self._adim
        wns = self.wavenumbers
        _, comp_curvs = self._egy_par_vals(spectrum)

        curv_moduli, curv_phases = np.abs(comp_curvs), np.angle(comp_curvs)
        prop_phases = wns * self.length + curv_phases

        kcm2 = self.ice.attenuations**2 + wns**2
        kcm2_diff = self.ice.attenuations**2 - wns**2
        st_2_sq = 2 * self.ice._red_elas_number**2
        st_2_sq2 = st_2_sq**2

        decp = self.ice.attenuations + self.ice._red_elas_number
        decm = self.ice.attenuations - self.ice._red_elas_number

        bigkpp = kcm2 + 2 * self.ice._red_elas_number * decp
        bigkpm = kcm2 - 2 * self.ice._red_elas_number * decm
        bigkmp = kcm2_diff + 2 * self.ice._red_elas_number * decp
        bigkmm = kcm2_diff - 2 * self.ice._red_elas_number * decm

        stk = self.ice._red_elas_number * wns

        edp = np.exp(-decp * self.length)
        edm = np.exp(-decm * self.length)

        q_base = (kcm2**2 - st_2_sq**2) ** 2 + 4 * st_2_sq**2 * kcm2_diff**2
        qdec_p = bigkpp**2 - (2 * stk) ** 2
        qdec_m = bigkpm**2 - (2 * stk) ** 2

        coefs = self._dis_hom_coefs(spectrum)
        x1, x2 = map(np.real, coefs[[0, 2]])
        y1, y2 = map(np.imag, coefs[[0, 2]])

        return (
            -2
            * st_2_sq
            * (
                y1 * int_cosh_cos()
                + x1 * int_sinh_sin()
                + y2 * int_sinh_cos()
                + x2 * int_cosh_sin()
            )
        )

    def energy(self, spectrum):
        return (
            self.ice.flex_rigidity
            * (
                self._egy_hom(spectrum)
                + 2 * self._egy_m(spectrum)
                + self._egy_par(spectrum)
            )
            / (2 * self.ice.thickness)
        )


class DiscreteSpectrum:
    def __init__(self, amplitudes, frequencies, phases=0, betas=0):

        # np.ravel force precisely 1D-arrays
        # Promote the map to list so the iterator can be used several times
        args = list(map(np.ravel, (amplitudes, frequencies, phases, betas)))
        (size,) = np.broadcast_shapes(*(arr.shape for arr in args))

        # TODO: sort waves by frequencies or something
        # TODO: sanity checks on nan, etc. that could be returned
        #       by the Spectrum objects

        if size != 1:
            for i, arr in enumerate(args):
                if arr.size == 1:
                    args[i] = itertools.repeat(arr[0], size)

        self.__waves = [
            Wave(_a, frequency=_f, phase=_ph, beta=_b) for _a, _f, _ph, _b in zip(*args)
        ]

    @property
    def waves(self):
        return self.__waves

    @functools.cached_property
    def _ang_freq2(self):
        return np.asarray([wave.angular_frequency2 for wave in self.waves])

    @functools.cached_property
    def _amps(self):
        return np.asarray([wave.amplitude for wave in self.waves])

    @functools.cached_property
    def _phases(self):
        return np.asarray([wave.phase for wave in self.waves])


class _GenericSpectrum:
    def __init__(
        self,
        *,
        u=None,
        swh=None,
        peak_period=None,
        peak_frequency=None,
        peak_wavelength=None,
    ):
        # if u is not None:
        #     values = SpecVars(u)
        #     for k, v in zip(
        #         (
        #             "swh",
        #             "peak_period",
        #             "peak_frequency",
        #             "peak_wavenumber",
        #             "peak_wavelength",
        #         ),
        #         values,
        #     ):
        #         setattr(self, f"__{k}", v)
        # else:
        ...

    @property
    def waves(self):
        return self.__waves


# def SpecVars(u=10, v=False):
#     Tp = 2 * np.pi * u / (0.877 * g)
#     fp = 1 / Tp
#     Hs = 0.0246 * u**2
#     k = (2 * np.pi * fp)**2 / g
#     wl = 2 * np.pi / k

#     if v:
#         print(f'For winds of {u:.2f}m/s, expected waves are {Hs:.2f}m high,\n'
#               f'with a period of {Tp:.2f}s, '
#               f'corresponding to a frequency of {fp:.2f}Hz,\n'
#               f'and wavenumber of {k:.2f}/m or wavelength of {wl:.2f}m.')
#     else:
#         return(Hs, Tp, fp, k, wl)


class PiersonMoskowitz(_GenericSpectrum):
    def __init__(
        self,
        *,
        wind_speed=None,
        swh=None,
        peak_frequency=None,
        peak_period=None,
        peak_wavelength=None,
    ):
        kwargs = locals()
        print(kwargs)
        kwargs.pop("self")
        keys = list(kwargs.keys())

        def filter_dict(dct):
            return (
                item[0]
                for item in filter(lambda item: item[1] is not None, kwargs.items())
            )

        if wind_speed is not None:
            self._init_from_u(kwargs.pop("wind_speed"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Wind speed was specified, {k} will be ignored", stacklevel=1
                )
        elif swh is not None:
            for k in keys[:1]:
                kwargs.pop(k)
            self._init_from_swh(kwargs.pop("swh"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Significant wave height was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_frequency is not None:
            for k in keys[:2]:
                kwargs.pop(k)
            self._init_from_frequency(kwargs.pop("peak_frequency"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Peak frequency was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_period is not None:
            for k in keys[:3]:
                kwargs.pop(k)
            self._init_from_period(kwargs.pop("peak_period"))
            for k in filter_dict(kwargs):
                warnings.warn(
                    f"Peak frequency was specified, {k} will be ignored",
                    stacklevel=1,
                )
        elif peak_wavelength is not None:
            for k in keys[:4]:
                kwargs.pop(k)
            self._init_from_frequency(kwargs.pop("peak_wavelength"))
        else:
            raise ValueError("At least one spectral parameter has to be provided")

    @property
    def wind_speed(self):
        return self.__wind_speed

    @property
    def swh(self):
        return self.__swh

    @property
    def peak_frequency(self):
        return self.__peak_frequency

    @property
    def peak_period(self):
        return self.__peak_period

    @property
    def peak_wavelength(self):
        return self.__peak_wavelength

    def _init_from_u(self, wind_speed):
        self.__wind_speed = wind_speed
        self.__swh = 0.0246 * wind_speed**2
        self.__peak_frequency = 0.877 * g / (2 * PI * wind_speed)
        self.__peak_period = 1 / self.peak_frequency

    def __call__(self, frequency):
        alpha_s = 0.2044
        beta_s = 1.2500

        frequency = np.asarray(frequency)

        return (
            alpha_s
            * self.swh**2
            * (self.peak_frequency**4 / frequency**5)
            * np.exp(-beta_s * (self.peak_frequency / frequency) ** 4)
        )


# class Spectrum:
#     def __init__(
#         self,
#         *,
#         u=None,
#         swh=None,
#         peak_period=None,
#         peak_frequency=None,
#         peak_wavelength=None,
#         beta=0,
#         phi=np.nan,
#         df=1.1,
#         x=-5,
#         y=16,
#         n=2,
#     ):
#         # if u is None or (
#         swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = SpecVars(
#             u
#         )  # Parameters for a typical spectrum
#         spec = SpecType
#         tfac = tail_fac

#         f = peak_frequency * df ** np.arange(x, y)

#         for key, value in kwargs.items():
#             if key == "u":
#                 u = value
#                 swh, peak_period, peak_frequency, peak_wavenumber, peak_wavelength = (
#                     SpecVars(u)
#                 )
#                 f = peak_frequency * df ** np.arange(x, y)
#             elif key == "Hs":
#                 swh = value
#             elif key == "n0":
#                 swh = (2**1.5) * value
#             elif key == "fp":
#                 peak_frequency = value
#                 peak_period = 1 / peak_frequency
#                 peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
#                 peak_wavelength = 2 * np.pi / peak_wavenumber
#             elif key == "Tp":
#                 peak_period = value
#                 peak_frequency = 1 / peak_period
#                 peak_wavenumber = (2 * np.pi * peak_frequency) ** 2 / g
#                 peak_wavelength = 2 * np.pi / peak_wavenumber
#             elif key == "wlp":
#                 peak_wavelength = value
#                 peak_wavenumber = 2 * np.pi / peak_wavelength
#                 peak_frequency = (g * peak_wavenumber) ** 0.5 / (2 * np.pi)
#                 peak_period = 1 / peak_frequency
#             elif key == "beta":
#                 beta = value
#             elif key == "phi":
#                 phi = value
#             elif key == "df":
#                 df = value
#                 fac = np.log(1.1) / np.log(df)
#                 x = -np.ceil(5 * fac)
#                 y = np.ceil(15 * fac) + 1
#             elif key == "spec" or key == "SpecType":
#                 spec = value
#             elif key == "f":
#                 f = value
#             elif key == "n":
#                 n = value
#             elif key == "tail_fac":
#                 tfac = value
#             else:
#                 print(f"Unknow input: {key}")

#         self.type = "WaveSpec"
#         self.SpecType = spec
#         self.Hs = swh
#         self.Tp = peak_period
#         self.fp = peak_frequency
#         self.kp = peak_wavenumber
#         self.wlp = peak_wavelength
#         self.beta = beta
#         self.tail_fac = tfac

#         if len(f) == 1 or spec == "Mono":
#             self.f = np.array([peak_frequency])
#             df_vec = np.array([1])
#             self.Ei = np.array([swh**2 / 16])
#         elif spec == "PowerLaw":
#             if n > 0:
#                 self.f = peak_frequency * df ** (np.arange(-f.size, 0) + 1)
#             else:
#                 self.f = peak_frequency * df ** (np.arange(0, f.size) + 1)
#             df_vec = np.empty_like(f)
#             df_vec[0] = f[1] - f[0]
#             df_vec[1:-1] = (f[2:] - f[:-2]) / 2
#             df_vec[-1] = f[-1] - f[-2]
#             self.Ei = PowerLaw(swh, peak_frequency, self.f, df_vec, n)
#         else:
#             self.f = f
#             df_vec = np.empty_like(f)
#             df_vec[0] = f[1] - f[0]
#             df_vec[1:-1] = (f[2:] - f[:-2]) / 2
#             df_vec[-1] = f[-1] - f[-2]
#             if spec == "JONSWAP":
#                 self.Ei = Jonswap(swh, peak_frequency, f)
#             elif spec == "PM":
#                 self.Ei = PM(u, f)
#             else:
#                 raise ValueError(f"Unknown spectrum type: {spec}")

#         self.nf = f.size
#         self.k = (2 * np.pi * self.f) ** 2 / g
#         self.cgw = 0.5 * (g / self.k) ** 0.5
#         self.nf = len(self.f)
#         self.df = df_vec

#         if type(phi) == np.ndarray:
#             self.phi = phi
#         else:
#             self.phi = phi * np.ones_like(self.f)

#         self.setWaves()
#         self.af = [0] * self.nf


class OceanCoupled(Ocean):
    """Extend `Ocean` to include wave-dependent quantities."""

    def __init__(self, ocean: Ocean, ds: DiscreteSpectrum, gravity: float):
        super().__init__(ocean.depth, ocean.density)
        self.__wavenumbers = self.compute_wavenumbers(ds, gravity)

    @property
    def wavenumbers(self) -> np.ndarray:
        """Array of wave numbers in rad m**-1"""
        return self.__wavenumbers

    @functools.cached_property
    def wavelengths(self) -> np.ndarray:
        """Wavelengths in m"""
        return 2 * PI / self.wavenumbers

    def _comp_wns(self, angfreqs2: np.ndarray, gravity: float) -> np.ndarray:
        def f(kk: float, alpha: float) -> float:
            # Dispersion relation (form f(k) = 0), for a free surface,
            # admitting one positive real root.
            return kk * np.tanh(kk) - alpha

        def df_dk(kk: float, alpha: float) -> float:
            # Derivative of dr with respect to kk.
            tt = np.tanh(kk)
            return tt + kk * (1 - tt**2)

        alphas = angfreqs2 / gravity
        if np.isposinf(self.depth):
            return alphas

        coefs_d0 = alphas * self.depth
        roots = np.full(len(coefs_d0), np.nan)
        for i, _d0 in enumerate(coefs_d0):
            if _d0 >= np.arctanh(np.nextafter(1, 0)):
                roots[i] = _d0
                continue
            res = optimize.root_scalar(f, (_d0,), fprime=df_dk, x0=_d0)
            if not res.converged:
                warnings.warn(
                    f"Root finding did not converge: free surface, "
                    f"f={np.sqrt(_d0/self.depth*gravity)/(2*PI):1.2g} Hz",
                    stacklevel=2,
                )
            roots[i] = res.root

        return roots / self.depth

    def compute_wavenumbers(self, ds: DiscreteSpectrum, gravity: float) -> np.ndarray:
        """ """
        return self._comp_wns(
            np.array([wave.angular_frequency2 for wave in ds.waves]), gravity
        )


class WaveSpectrum:
    pass


class Domain:
    """"""

    def __init__(
        self,
        spectrum: WaveSpectrum,
        ocean: Ocean,
        nf: int,
        phases,
        betas,
        fmin,
        fmax,
        frequencies,
        gravity,
    ) -> None:
        """"""
        self.__spectrum = spectrum
        self.__ocean = OceanCoupled(ocean, spectrum)

        self.__frozen_spectrum = DiscreteSpectrum(
            spectrum.amplitude(frequencies), frequencies, phases, betas
        )
        # TODO: doit avoir un attribut gravité si le spectre n'en a pas

    def _init_from_f(self): ...

    def add_floes(self, floes):
        # TODO: test sur l'attribut ice des floes, ajouter l'IceCoupled
        # correspondant en attribut des FloeCoupled, mais stocker les
        # IceCoupled dans Domain pour centralisation et éviter de recalculer
        # les wavenumbers, etc.: les FloeCoupled ne devraient qu'en avoir une
        # référence
        # TODO: les phases de chaque floe peuvent également être déterminée
        # ici, à la chaine
        ...
