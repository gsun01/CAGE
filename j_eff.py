from __future__ import annotations
import numpy as np
from typing import Callable, Literal, Tuple

__all__ = [
    "j_eff"
]

Polarization = Literal["+", "x"]

def _asarray_f64(x):
    return np.asarray(x, dtype=float)

def f(x) -> float:
    return -3 - 6*1j/x - 12*np.exp(-1j*x)/(x**2) - 12*1j*(1-np.exp(-1j*x))/(x**3)

def j_eff(pol:Polarization, wg:float) -> Tuple[Callable, Callable, Callable]:
    if pol == "+":  # plus polarization
        def Jr_plus(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            fwz = f(wg*z)
            return fwz * np.sin(2*phi) * r
        def Jphi_plus(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            fwz = f(wg*z)
            return fwz * np.cos(2*phi) * r
        def Jz_plus(r, phi, z):
            z = _asarray_f64(z)
            return np.zeros_like(z, dtype=complex)
        return Jr_plus, Jphi_plus, Jz_plus
    
    else:  # cross polarization
        def Jr_cross(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            fwz = f(wg*z)
            return -fwz * np.cos(2*phi) * r
        def Jphi_cross(r, phi, z):
            r, phi, z = _asarray_f64(r), _asarray_f64(phi), _asarray_f64(z)
            fwz = f(wg*z)
            return -fwz * np.sin(2*phi) * r
        def Jz_cross(r, phi, z):
            z = _asarray_f64(z)
            return np.zeros_like(z, dtype=complex)
        return Jr_cross, Jphi_cross, Jz_cross