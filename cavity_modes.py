# from scipy.special import jv, jvp
# import numpy as np
# from bessel import initialize_bessel_table, BesselJZeros, BesselJpZeros

# def m_over_r_Jm(m, gamma, r):
#     # safe m/r with no 1/r singularity
#     return 0.5*gamma*(jv(m-1, gamma*r) + jv(m+1, gamma*r))

# def k_z(p, L):
#     return p*np.pi/L

# def gamma_mn(family, m, n, a):
#     if family.upper() == "TM":
#         return BesselJZeros(m, n)/a
#     elif family.upper() == "TE":
#         return BesselJpZeros(m, n)/a
#     else:
#         raise ValueError("family must be 'TE' or 'TM'")

# def omega_mnp(gamma_mn, kz_p):
#     return np.sqrt(gamma_mn**2 + kz_p**2)

# # def mnp_guard(m, n, p):
# #     if not (isinstance(m, int) and isinstance(n, int) and isinstance(p, int)):
# #         raise ValueError("m, n, p must be integers")
# #     if m < 0 or n <= 0 or p < 0:
# #         raise ValueError("m must be >= 0, n must be > 0, p must be >= 0")

# # TM mode functions
# def Er_TM_fn(m, n, p, a, L, sign="+"):
#     g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
#     def f(r, phi, z):
#         ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
#         return -(kz/g) * ang * np.sin(kz*z) * jvp(m, g*r)
#     return f

# def Ephi_TM_fn(m, n, p, a, L, sign="+"):
#     g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
#     def f(r, phi, z):
#         ang = np.cos(m*phi) if sign == "+" else -np.sin(m*phi)
#         return -(kz/g**2) * ang * np.sin(kz*z) * m_over_r_Jm(m, g, r)
#     return f

# def Ez_TM_fn(m, n, p, a, L, sign="+"):
#     g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
#     def f(r, phi, z):
#         ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
#         return ang * np.cos(kz*z) * jv(m, g*r)
#     return f

# # TE mode functions
# def Er_TE_fn(m, n, p, a, L, sign="+", c=1.0):
#     g  = gamma_mn("TE", m, n, a); kz = k_z(p, L); w = omega_mnp(g, kz)
#     def f(r, phi, z):
#         ang = np.cos(m*phi) if sign == "+" else -np.sin(m*phi)
#         return 1j * w/g**2 * ang * np.sin(kz*z) * m_over_r_Jm(m, g, r)
#     return f

# def Ephi_TE_fn(m, n, p, a, L, sign="+", c=1.0):
#     g  = gamma_mn("TE", m, n, a); kz = k_z(p, L); w = omega_mnp(g, kz)
#     def f(r, phi, z):
#         ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
#         return -1j * w/g * ang * np.sin(kz*z) * jvp(m, g*r)
#     return f

# def Ez_TE_fn(m, n, p, a, L, sign="+"):
#     def f(r, phi, z):
#         # TE has Ez = 0 in this convention
#         return np.zeros_like(np.asarray(r, dtype=float))
#     return f


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
        Speed of light factor for unit systems where you want it explicit.
        Only appears as a multiplier in TE amplitudes here; set c=1 for natural units.
    init_bessel : dict | None
        If provided, passed to `initialize_bessel_table(**init_bessel)` once.
        For example: dict(m_max=20, n_max=20).
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

    # Convenience wrappers matching your original naming
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