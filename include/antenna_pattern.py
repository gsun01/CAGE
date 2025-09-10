from __future__ import annotations
__all__ = ["antenna_pattern"]
from custom_types import *
import numpy as np
from j_eff import j_eff_vectorized

def antenna_pattern(
    r:NDArray, phi:NDArray, z:NDArray,
    Er3:NDArray, Ephi3:NDArray, Ez3:NDArray,
    *,
    pol: Polarization, wg: float,
    a: float, L: float,
    beta:NDArray = asarr_f64(0.0), alpha:NDArray = asarr_f64(0.0), psi:NDArray = asarr_f64(0.0),
    savefile: str | None = None
) -> NDArray:
    """
    Compute η(β, α, ψ) where both the cavity mode and j_eff are provided as vectorized components.

    Parameters
    ----------
    r, phi, z : 1D arrays
        Cylindrical grid points (r in [0,a], phi in [0,2π), z in [0,L]).
    Er, Ephi, Ez : vectorized cavity mode components
    pol : {'+','x'm 'plus', 'cross'}
    wg : f64
        GW wavenumber used inside j_eff_off_axis.
    a, L : f64
        Cavity radius and length.
    beta, alpha, psi : 1D arrays
        Angle grids. beta in [0,π], alpha in [0,2π), psi in [0,2π).

    Returns
    -------
    eta : ndarray
        Shape (Nbeta, Nalpha) if reduced, else (Nbeta, Nalpha, Npsi).
    """

    # do E^2 integral on (R,PHI,Z) grid to get a scalar norm
    R3, PHI3, Z3 = np.meshgrid(r, phi, z, indexing='ij', sparse=True)
    Vcav = np.pi * a**2 * L
    E2 = np.abs(Er3)**2 + np.abs(Ephi3)**2 + np.abs(Ez3)**2
    if not np.isfinite(E2).all() or np.max(E2) == 0.0:
        raise ValueError("Mode callables produced non-finite or all-zero fields.")
    norm = np.sqrt(Vcav * np.trapezoid(np.trapezoid(np.trapezoid(E2*R3, z, axis=2), phi, axis=1), r, axis=0))
    print(f"Norm is {norm}")

    # Now broadcast to 6D grid: shape (Nr, Nphi, Nz, Nbeta, Nalpha, Npsi)
    R6, PHI6, Z6, BETA6, ALPHA6, PSI6 = np.meshgrid(r, phi, z, beta, alpha, psi, 
                                                    indexing='ij', sparse=True)
    grid_shape = R6.shape
    # extend Er,Ephi,Ez to have trailing singleton dim for broadcasting with jr,jphi,jz
    Er6, Ephi6, Ez6 = Er3[...,None,None,None], Ephi3[...,None,None,None], Ez3[...,None,None,None]
    print(f"E shape after broadcasting: {Er6.shape}")
    
    print("Calculating j_eff...")
    jr, jphi, jz = j_eff_vectorized(R6, PHI6, Z6, BETA6, ALPHA6, PSI6, pol=pol, wg=wg)
    print(f"j shape after broadcasting: {jr.shape}")
    
    E_dot_j = (np.conj(Er6)*jr + np.conj(Ephi6)*jphi + np.conj(Ez6)*jz)
    integrand = E_dot_j * R6  # include Jacobian r
    print("Finished j_eff calculation, starting integral...")
    I = np.trapezoid(np.trapezoid(np.trapezoid(integrand, z, axis=2), phi, axis=1), r, axis=0)

    print("Finished integral, starting eta calculation...")

    eta = np.divide(np.abs(I), norm, out=np.zeros_like(I, dtype=f64), where=norm!=0.0)

    print("Finished eta calculation.")

    if savefile is not None:
        np.save(savefile, eta)
        print(f"Saved eta to {savefile}")
    return eta
