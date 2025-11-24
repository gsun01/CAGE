from __future__ import annotations
__all__ = ["antenna_pattern"]

import numpy as np
from pathlib import Path

from include.custom_types import *
from include.j_eff_from_ana import compute_j_eff_6d

def antenna_pattern(
    R3:NDArray, PHI3:NDArray, Z3:NDArray,
    BETA3:NDArray, ALPHA3:NDArray, PSI3:NDArray,
    Er3:NDArray, Ephi3:NDArray, Ez3:NDArray, 
    norm: f64, wg: f64,
    kernel_path: str | Path,
    eta_p_save_path: str | Path,
    eta_c_save_path: str | Path,
    j_eff_save_dir: str | Path | None = None
) -> tuple[NDArray, NDArray]:
    """
    Compute eta(beta, alpha, psi) on 6D spatial+angle grid given cavity modes and j_eff kernels.

    Parameters
    ----------
    R3, PHI3, Z3 : NDArray
        Shape (Nr, Nphi, Nz) grid of spatial coordinates.
    BETA3, ALPHA3, PSI3 : ndarray
        Shape (Nbeta, Nalpha, Npsi) grid of angles.
    Er3, Ephi3, Ez3 : NDArray
        Shape (Nr, Nphi, Nz) cavity mode field components.
    norm, wg : float
        Normalization factor and angular frequency of the cavity mode.
    kernel_path : str or os.PathLike
        Path to the j_eff kernel pickle file.
    eta_p_save_path, eta_c_save_path : str or os.PathLike
        Paths to save the computed eta arrays.
    j_eff_save_dir : str or os.PathLike, optional
        Directory to save computed j_eff arrays. If None, j_eff arrays are not saved.

    Returns
    -------
    (eta_p, eta_c) : tuple[NDArray, NDArray]
        Tuple of two eta arrays, each of shape (Nbeta, Nalpha, Npsi).
    """
    # extend R,Er,Ephi,Ez to have trailing singleton dim for broadcasting with jr,jphi,jz
    R6 = R3[...,None,None,None]
    Er6, Ephi6, Ez6 = Er3[...,None,None,None], Ephi3[...,None,None,None], Ez3[...,None,None,None]
    
    # print("Calculating j_eff...", flush=True)
    Jr_p6, Jphi_p6, Jz_p6, Jr_c6, Jphi_c6, Jz_c6 = compute_j_eff_6d(
        R3, Z3, PHI3,
        BETA3, ALPHA3, PSI3,
        wg,
        kernel_path,
        save_dir=j_eff_save_dir
    )

    # print("Calculating int(E dot J)...", flush=True)
    integrand_p = (np.conj(Er6)*Jr_p6 + np.conj(Ephi6)*Jphi_p6 + np.conj(Ez6)*Jz_p6) * R6  # include Jacobian r
    integrand_c = (np.conj(Er6)*Jr_c6 + np.conj(Ephi6)*Jphi_c6 + np.conj(Ez6)*Jz_c6) * R6

    # use 1D coordinate arrays for safe integration
    r = R3[:,0,0]
    phi = PHI3[0,:,0]
    z = Z3[0,0,:]
    I_p = np.trapezoid(np.trapezoid(np.trapezoid(integrand_p, z, axis=2), phi, axis=1), r, axis=0)
    I_c = np.trapezoid(np.trapezoid(np.trapezoid(integrand_c, z, axis=2), phi, axis=1), r, axis=0)

    # print("Starting eta calculation...", flush=True)
    # norm includes the V^1/2 factor
    eta_p = np.divide(np.abs(I_p), norm, out=np.zeros_like(I_p, dtype=np.float64), where=norm!=0.0)
    eta_c = np.divide(np.abs(I_c), norm, out=np.zeros_like(I_c, dtype=np.float64), where=norm!=0.0)
    # print("Finished eta calculation.", flush=True)

    if eta_p_save_path is not None:
        try:
            eta_p_save_path = Path(eta_p_save_path)
            np.save(eta_p_save_path, eta_p)
            print(f"Saved eta to {eta_p_save_path}", flush=True)
        except Exception as e:
            print(f"Failed to save eta to {eta_p_save_path}: {e}", flush=True)
    if eta_c_save_path is not None:
        try:
            eta_c_save_path = Path(eta_c_save_path)
            np.save(eta_c_save_path, eta_c)
            print(f"Saved eta to {eta_c_save_path}", flush=True)
        except Exception as e:
            print(f"Failed to save eta to {eta_c_save_path}: {e}", flush=True)
    return eta_p, eta_c
