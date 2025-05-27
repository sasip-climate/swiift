import functools
import warnings

import attrs
import numpy as np
import scipy.optimize as optimize

# TODO: add from_ocean, from_ice class methods, use them model.Domain where
# relevant


@attrs.define
class FreeSurfaceSolver:
    alphas: np.ndarray
    depth: float

    def f(self, kk: float, alpha: float) -> float:
        # Dispersion relation (form f(k) = 0), for a free surface,
        # admitting one positive real root.
        return kk * np.tanh(kk) - alpha

    def df_dk(self, kk: float, alpha: float) -> float:
        # Derivative of dr with respect to kk.
        tt = np.tanh(kk)
        return tt + kk * (1 - tt**2)

    def find_k(self, k0, alpha):
        res = optimize.root_scalar(self.f, args=(alpha,), fprime=self.df_dk, x0=alpha)
        if not res.converged:
            warnings.warn(
                "Root finding did not converge: free surface",
                stacklevel=2,
            )
        return res.root

    def compute_wavenumbers(self) -> np.ndarray:
        if np.isposinf(self.depth):
            return self.alphas

        coefs_d0 = self.alphas * self.depth
        roots = np.full(len(coefs_d0), np.nan)
        for i, _d0 in enumerate(coefs_d0):
            if _d0 >= np.arctanh(np.nextafter(1, 0)):
                roots[i] = _d0
                continue

            find_k_i = functools.partial(
                self.find_k,
                alpha=_d0,
            )
            roots[i] = find_k_i(_d0)

        return roots / self.depth


@attrs.define
class ElasticMassLoadingSolver:
    alphas: np.ndarray
    deg1: np.ndarray
    deg0: np.ndarray
    scaled_ratio: float

    def f(self, kk: float, d0: float, d1: float, rr: float) -> float:
        return (kk**5 + d1 * kk) * np.tanh(rr * kk) + d0

    def df_dk(self, kk: float, d0: float, d1: float, rr: float) -> float:
        return (5 * kk**4 + d1 + rr * d0) * np.tanh(rr * kk) + rr * (kk**5 + d1 * kk)

    def extract_real_root(self, roots):
        mask = (np.imag(roots) == 0) & (np.real(roots) > 0)
        if mask.nonzero()[0].size != 1:
            raise ValueError("An approximate initial guess could not be found")
        return np.real(roots[mask][0])

    def find_k(self, k0, alpha, d0, d1, rr):
        res = optimize.root_scalar(
            self.f,
            args=(d0, d1, rr),
            fprime=self.df_dk,
            x0=k0,
            xtol=1e-10,
        )
        if not res.converged:
            warnings.warn(
                "Root finding did not converge: ice-covered surface",
                stacklevel=2,
            )
        return res.root

    def compute_wavenumbers(self) -> np.ndarray:
        roots = np.full(self.alphas.size, np.nan)

        for i, (alpha, _d0, _d1) in enumerate(zip(self.alphas, self.deg0, self.deg1)):
            find_k_i = functools.partial(
                self.find_k,
                alpha=alpha,
                d0=_d0,
                d1=_d1,
                rr=self.scaled_ratio,
            )

            # We always expect one positive real root,
            # and if _d1 < 0, eventually two additional negative real roots.
            roots_dw = np.polynomial.polynomial.polyroots([_d0, _d1, 0, 0, 0, 1])
            k0_dw = self.extract_real_root(roots_dw)
            if np.isposinf(self.scaled_ratio):
                roots[i] = k0_dw
                continue
            # Use a DW initial guess if |1-1/tanh(rr*k_DW)| < 0.15
            # Use a SW initial guess if |1-rr*k_SW/tanh(rr*k_SW)| < 0.20
            thrsld_dw, thrsld_sw = 1.33, 0.79
            if self.scaled_ratio * k0_dw > thrsld_dw:
                roots[i] = find_k_i(k0_dw)
            else:
                roots_sw = np.polynomial.polynomial.polyroots(
                    [_d0 / self.scaled_ratio, 0, _d1, 0, 0, 0, 1]
                )
                k0_sw = self.extract_real_root(roots_sw)

                if self.scaled_ratio * k0_sw < thrsld_sw:
                    roots[i] = find_k_i(k0_sw)
                # Use an initial guess in the middle otherwise
                else:
                    k0_ = (k0_sw + k0_dw) / 2
                    roots[i] = find_k_i(k0_)

        return roots
