from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal, Tuple

import numpy as np
from scipy.special import jv, jvp

# Your bessel helper provides these; initialize once if you like.
from bessel import initialize_bessel_table, BesselJZeros, BesselJpZeros

__all__ = ["CavityModeHelper"]

Family = Literal["TE", "TM"]
Sign = Literal["+", "-"]

def _asarray_f64(x):
    return np.asarray(x, dtype=float)

def _m_over_r_Jm(m: int, gamma: float, r) -> np.ndarray:
    """Safe m/r with no 1/r singularity, using the Bessel identity."""
    r = _asarray_f64(r)
    return 0.5 * gamma * (jv(m - 1, gamma * r) + jv(m + 1, gamma * r))

@dataclass(frozen=True)
class CavityModeHelper:
    """
    Helper for cylindrical cavity eigenmodes (TE/TM) in (r, φ, z).

    Parameters
    ----------
    a : float
        Cylinder radius.
    L : float
        Cylinder length.
    c : float, optional
        Speed of light.
    """
    a: float
    L: float
    c: float = 1.0

    def __post_init__(self):
        if not (self.a > 0 and self.L > 0):
            raise ValueError("a and L must be positive.")

    # -------- core dispersion helpers --------

    @staticmethod
    def _kz(p: int, L: float) -> float:
        if not (isinstance(p, int) and p >= 0):
            raise ValueError("p must be an integer >= 0")
        return p * np.pi / L

    @lru_cache(maxsize=None)
    def _gamma(self, family: Family, m: int, n: int) -> float:
        fam = family.upper()
        if not (isinstance(m, int) and isinstance(n, int)):
            raise ValueError("m and n must be integers")
        if m < 0 or n <= 0:
            raise ValueError("m >= 0 and n > 0 are required")

        if fam == "TM":
            return float(BesselJZeros(m, n) / self.a)
        elif fam == "TE":
            return float(BesselJpZeros(m, n) / self.a)
        else:
            raise ValueError("family must be 'TE' or 'TM'")

    @staticmethod
    def _omega(gamma: float, kz: float) -> float:
        return float(np.sqrt(gamma * gamma + kz * kz))

    # -------- public API --------

    def mode_functions(
        self, family: Family, m: int, n: int, p: int, sign: Sign = "+"
    ) -> Tuple[Callable, Callable, Callable]:
        """
        Return callables (Er, Ephi, Ez) for the requested (family, m, n, p).

        Each callable has signature f(r, phi, z) and supports NumPy broadcasting.
        """
        fam = family.upper()
        if sign not in {"+", "-"}:
            raise ValueError("sign must be '+' or '-'")

        g = self._gamma(fam, m, n)
        kz = self._kz(p, self.L)
        w = self._omega(g, kz)

        if fam == "TM":
            def Er_TM(r, phi, z):
                r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
                ang = np.sin(m * phi) if sign == "+" else np.cos(m * phi)
                return -(kz / g) * ang * np.sin(kz * z) * jvp(m, g * r)

            def Ephi_TM(r, phi, z):
                r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
                ang = np.cos(m * phi) if sign == "+" else -np.sin(m * phi)
                return -(kz / (g * g)) * ang * np.sin(kz * z) * _m_over_r_Jm(m, g, r)

            def Ez_TM(r, phi, z):
                r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
                ang = np.sin(m * phi) if sign == "+" else np.cos(m * phi)
                return ang * np.cos(kz * z) * jv(m, g * r)

            return Er_TM, Ephi_TM, Ez_TM

        # TE family
        def Er_TE(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            ang = np.cos(m * phi) if sign == "+" else -np.sin(m * phi)
            # include self.c for unit bookkeeping if desired
            return 1j * (self.c * w) / (g * g) * ang * np.sin(kz * z) * _m_over_r_Jm(m, g, r)

        def Ephi_TE(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            ang = np.sin(m * phi) if sign == "+" else np.cos(m * phi)
            return -1j * (self.c * w) / g * ang * np.sin(kz * z) * jvp(m, g * r)

        def Ez_TE(r, phi, z):
            r = _asarray_f64(r)
            # TE has Ez = 0 in this convention
            return np.zeros_like(r)

        return Er_TE, Ephi_TE, Ez_TE

    # wrappers for individual components
    def Er_TM_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TM", m, n, p, sign)[0]

    def Ephi_TM_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TM", m, n, p, sign)[1]

    def Ez_TM_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TM", m, n, p, sign)[2]

    def Er_TE_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TE", m, n, p, sign)[0]

    def Ephi_TE_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TE", m, n, p, sign)[1]

    def Ez_TE_fn(self, m: int, n: int, p: int, sign: Sign = "+") -> Callable:
        return self.mode_functions("TE", m, n, p, sign)[2]