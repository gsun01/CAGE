"""
Routine for computing 6D effective current (j_eff) grids
from pickled SymPy expressions using spatial-grid chunking 
for CPU cache efficiency.

This version is single-threaded and computes all 6 components serially.
q-field and Fk maps are calculated on-the-fly inside the spatial loop
to avoid memory-intensive 6D arrays.
"""

from __future__ import annotations
__all__ = ["compute_j_eff_6d"]

import numpy as np
import pickle
import sympy as sp
from pathlib import Path

from include.custom_types import *

# --- F_k(q) Functions ---
# small-q expansions included up to constant terms
def F1_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    result = -1j/(2*q_safe) - np.exp(-1j*q_safe)/q_safe**2 - 1j*(1-np.exp(-1j*q_safe))/q_safe**3
    return np.where(q == 0, 1.0/3.0 - 0.0j, result)

def F2_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    result = -(1+np.exp(-1j*q_safe))/q_safe**2 - 2j*(1-np.exp(-1j*q_safe))/q_safe**3
    return np.where(q == 0, 1.0/6.0 - 0.0j, result)

def F2p_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    exp_term = np.exp(-1j*q_safe)
    one_minus_exp = (1 - exp_term)
    result = (1j*exp_term/q_safe**2 
            - 2*(-1 - exp_term)/q_safe**3 
            + 2.0*exp_term/q_safe**3 
            + 6.0*1j*one_minus_exp/q_safe**4)
    return np.where(q == 0, 0.0 - 1.0j/12.0, result)

# --- Helpers ---
def pol_tensor_6d(BETA3: NDArray, ALPHA3: NDArray, PSI3: NDArray) -> NDArray:
    ca, sa = np.cos(ALPHA3), np.sin(ALPHA3)
    cb, sb = np.cos(BETA3),  np.sin(BETA3)
    c2a, s2a = np.cos(2*ALPHA3), np.sin(2*ALPHA3)
    c2p, s2p = np.cos(2*PSI3),   np.sin(2*PSI3)
    Axx = cb*cb*ca*ca - sa*sa; Axy = ca*sa*(1+cb*cb); Axz = -cb*sb*ca
    Ayy = cb*cb*sa*sa - ca*ca; Ayz = -cb*sb*sa; Azz = sb*sb
    Bxx = -cb * s2a; Bxy =  cb * c2a; Bxz =  sb * sa
    Byy =  cb * s2a; Byz = -sb * ca; Bzz = 0.0
    exx_p = c2p*Axx + s2p*Bxx; exy_p = c2p*Axy + s2p*Bxy; exz_p = c2p*Axz + s2p*Bxz
    eyy_p = c2p*Ayy + s2p*Byy; eyz_p = c2p*Ayz + s2p*Byz; ezz_p = c2p*Azz + s2p*Bzz
    exx_c = -s2p*Axx + c2p*Bxx; exy_c = -s2p*Axy + c2p*Bxy; exz_c = -s2p*Axz + c2p*Bxz
    eyy_c = -s2p*Ayy + c2p*Byy; eyz_c = -s2p*Ayz + c2p*Byz; ezz_c = -s2p*Azz + c2p*Bzz
    return np.stack([[exx_p, exy_p, exz_p, eyy_p, eyz_p, ezz_p], 
                     [exx_c, exy_c, exz_c, eyy_c, eyz_c, ezz_c]], 
                     axis=0, dtype=np.complex128)

def n_vec_6d(BETA3: NDArray, ALPHA3: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    sb, cb = np.sin(BETA3), np.cos(BETA3)
    ca, sa = np.cos(ALPHA3), np.sin(ALPHA3)
    nx = sb*ca; ny = sb*sa; nz = cb
    return nx, ny, nz

# --- New chunked helper functions ---
def _q_field_3d_slice(r_val: float, phi_val: float, z_val: float, 
                      BETA3: NDArray, ALPHA3: NDArray, wg: float) -> NDArray:
    """Calculates a 3D angular slice of the q-field."""
    sb, cb = np.sin(BETA3), np.cos(BETA3)
    # r,phi,z are scalars; BETA3, ALPHA3 are 3D angular grids
    return wg * ( r_val * sb * np.cos(phi_val - ALPHA3) + z_val * cb )

def _precalculate_fk_map_3d_slice(q_slice: NDArray) -> dict:
    """Calculates the Fk map for a 3D angular slice of q."""
    Fk_slice_map = {}
    q_slice_sq = q_slice * q_slice
    F1_s = F1_func(q_slice); Fk_slice_map[0] = F1_s; Fk_slice_map[1] = q_slice * F1_s; del F1_s
    F2_s = F2_func(q_slice); Fk_slice_map[2] = F2_s; Fk_slice_map[3] = q_slice * F2_s; del F2_s
    F2p_s = F2p_func(q_slice); Fk_slice_map[4] = F2p_s; Fk_slice_map[5] = q_slice * F2p_s; Fk_slice_map[6] = q_slice_sq * F2p_s; del F2p_s, q_slice_sq
    return Fk_slice_map

# --- Core Routines ---

def load_kernels(path: str | Path) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """
    Loads pickled SymPy expressions and lambdifies them using the
    'numpy' backend with an explicit modules dictionary.
    """
    # print(f"Loading and compiling kernels from: {path}", flush=True)
    
    with open(path, "rb") as f:
        pack = pickle.load(f)
        
    var_names = pack["vars"]
    arg_symbols = sp.symbols(' '.join(var_names))

    numpy_modules = {
        'numpy': np,
        'I': 1j,
        'pi': np.pi
    }

    def lam(m):
        return {k: sp.lambdify(arg_symbols, v, modules=[numpy_modules], cse=True)
                for k, v in m.items()}
    
    kernels_sym = (
        pack["JR_plus_exprs"],
        pack["JPHI_plus_exprs"],
        pack["JZ_plus_exprs"],
        pack["JR_cross_exprs"],
        pack["JPHI_cross_exprs"],
        pack["JZ_cross_exprs"]
    )
    
    kernels_lam = (
        lam(pack["JR_plus_exprs"]),
        lam(pack["JPHI_plus_exprs"]),
        lam(pack["JZ_plus_exprs"]),
        lam(pack["JR_cross_exprs"]),
        lam(pack["JPHI_cross_exprs"]),
        lam(pack["JZ_cross_exprs"])
    )
    
    # print(f"Kernel compilation complete (NumPy backend). Args: {var_names}", flush=True)
    return kernels_sym, kernels_lam

def eval_component_grid_6d(
    coeff_map: dict, 
    coeff_map_sym: dict, 
    e6: NDArray, 
    R3: NDArray, PHI3: NDArray, Z3: NDArray, 
    nx: NDArray, ny: NDArray, nz: NDArray, 
    wg_val: float,
    BETA3: NDArray, 
    ALPHA3: NDArray
) -> NDArray:
    """
    Evaluates one component (Jr, Jphi, or Jz) on the full 6D grid
    by iterating over the spatial grid.
    """
    Nr, Nphi, Nz = R3.shape
    Nbeta, Nalpha, Npsi = nx.shape
    acc = np.zeros((Nr, Nphi, Nz, Nbeta, Nalpha, Npsi), dtype=c128)
    
    for i in range(Nr):
        for j in range(Nphi):
            for k in range(Nz):
                r_val = R3[i, j, k]
                phi_val = PHI3[i, j, k]
                z_val = Z3[i, j, k]

                # Calculate q and Fk for this 3D slice ONLY
                q_slice = _q_field_3d_slice(r_val, phi_val, z_val, BETA3, ALPHA3, wg_val)
                Fk_map_3d = _precalculate_fk_map_3d_slice(q_slice)
                
                acc_slice = np.zeros((Nbeta, Nalpha, Npsi), dtype=c128)
                
                for (term_j, term_k), func in coeff_map.items():
                    
                    try:
                        C_slice = func(r_val, phi_val, z_val, nx, ny, nz, q_slice, wg_val)
                        
                        # Fk_map_3d is a 3D dict, not 6D
                        Fk_slice = Fk_map_3d[term_k]
                        e6_slice = e6[term_j]
                        
                        term = C_slice * e6_slice * Fk_slice
                        acc_slice += term.astype(c128)
                        
                    except (TypeError, ValueError, NameError) as e:
                        print("\n\n" + "="*80)
                        print(f"CRITICAL ERROR: Failed to evaluate a symbolic expression term.")
                        print(f"Error: {e}")
                        print(f"Location: Spatial point (i={i}, j={j}, k={k}), Term (j={term_j}, k={term_k})")
                        
                        original_expr = coeff_map_sym.get((term_j, term_k))
                        print(f"\nProblematic Symbolic Expression (C_jk):\n{original_expr}")
                        
                        if hasattr(original_expr, 'free_symbols'):
                            print(f"\nFree symbols in this expression: {original_expr.free_symbols}") # type: ignore
                        
                        print(f"\nExpected symbols in lambdify: {list(coeff_map.values())[0].__code__.co_varnames}")
                        print("="*80 + "\n\n")
                        
                        raise e
                
                acc[i, j, k, :, :, :] = acc_slice
                
                # Manually delete large 3D slices to help garbage collector
                del q_slice, Fk_map_3d, acc_slice

    return acc

# --- Main Entry Point ---

def compute_j_eff_6d(
    # beta_str: str,
    R3: NDArray, PHI3: NDArray, Z3: NDArray,
    BETA3: NDArray, ALPHA3: NDArray, PSI3: NDArray,
    wg: float,
    kernel_path: str | Path,
    save_dir: str | Path | None = None
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    
    kernels_sym, kernels_lam = load_kernels(kernel_path)
    JR_plus_sym, JPHI_plus_sym, JZ_plus_sym, JR_cross_sym, JPHI_cross_sym, JZ_cross_sym = kernels_sym
    JR_plus_lam, JPHI_plus_lam, JZ_plus_lam, JR_cross_lam, JPHI_cross_lam, JZ_cross_lam = kernels_lam

    # print("Computing n-vectors, pol tensors, and q-field...", flush=True)
    nx, ny, nz = n_vec_6d(BETA3, ALPHA3)
    e6 = pol_tensor_6d(BETA3, ALPHA3, PSI3)
    e6_p, e6_c = e6[0], e6[1]

    # q and Fk_slice_map are now calculated inside eval_component_grid_6d

    results = {}
    print("--- Starting j_eff component computation ---", flush=True)

    try:
        # --- Plus Polarization ---
        # print("--- Starting evaluation: Jr_p ---", flush=True)
        results['Jr_p'] = eval_component_grid_6d(
            JR_plus_lam, JR_plus_sym, e6_p, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jr_p ---", flush=True)

        # print("--- Starting evaluation: Jphi_p ---", flush=True)
        results['Jphi_p'] = eval_component_grid_6d(
            JPHI_plus_lam, JPHI_plus_sym, e6_p, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jphi_p ---", flush=True)
        
        # print("--- Starting evaluation: Jz_p ---", flush=True)
        results['Jz_p'] = eval_component_grid_6d(
            JZ_plus_lam, JZ_plus_sym, e6_p, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jz_p ---", flush=True)

        # --- Cross Polarization ---
        # print("--- Starting evaluation: Jr_c ---", flush=True)
        results['Jr_c'] = eval_component_grid_6d(
            JR_cross_lam, JR_cross_sym, e6_c, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jr_c ---", flush=True)

        # print("--- Starting evaluation: Jphi_c ---", flush=True)
        results['Jphi_c'] = eval_component_grid_6d(
            JPHI_cross_lam, JPHI_cross_sym, e6_c, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jphi_c ---", flush=True)

        # print("--- Starting evaluation: Jz_c ---", flush=True)
        results['Jz_c'] = eval_component_grid_6d(
            JZ_cross_lam, JZ_cross_sym, e6_c, R3, PHI3, Z3, 
            nx, ny, nz, wg, BETA3, ALPHA3
        )
        # print("--- Finished component: Jz_c ---", flush=True)
    
    except Exception as exc:
        print(f'A component generated an exception: {exc}', flush=True)
        raise exc

    print("--- j_eff computation complete. ---", flush=True)

    # save results to disk
    if save_dir is not None:
        data_dir = Path(save_dir)
        try:
            # Need to get grid dimensions for filename
            Nr, Nphi, Nz = R3.shape
            Nbeta, Nalpha, Npsi = BETA3.shape
            
            data_dir.mkdir(parents=True, exist_ok=True)            
            for name, array in results.items():
                # file_path = data_dir / f"{name}_{Nr},{Nphi},{Nz},{Nbeta},{Nalpha},{Npsi}.npy"
                # test filename ##########################
                file_path = data_dir / f"{name}_{beta_str}.npy"
                np.save(file_path, array)
                print(f"    Saved {name} to {file_path.name}", flush=True)
            print(f"--- Saved results to: {data_dir.resolve()} ---", flush=True)
            
        except Exception as e:
            print(f"!!! FAILED TO SAVE RESULTS: {e} !!!", flush=True)

    return (results['Jr_p'], results['Jphi_p'], results['Jz_p'],
            results['Jr_c'], results['Jphi_c'], results['Jz_c'])

# --- Example Usage ---
if __name__ == "__main__":
    import sys
    
    print("Starting j_eff main execution...", flush=True)
    
    # 1. Define spatial grid
    Nr, Nphi, Nz = 30, 90, 50
    a, L = 0.0206, 0.2032

    r_ax = np.linspace(0, a, Nr)
    phi_ax = np.linspace(0, 2*np.pi, Nphi)
    z_ax = np.linspace(-L/2, L/2, Nz)
    R3, PHI3, Z3 = np.meshgrid(r_ax, phi_ax, z_ax, indexing='ij')

    # 2. Define angular grid
    Nbeta, Nalpha, Npsi = 1, 1, 1
    # beta_ax = np.linspace(0, np.pi, Nbeta)
    # alpha_ax = np.linspace(0, 2*np.pi, Nalpha)
    # psi_ax = np.linspace(0, 2*np.pi, Npsi)
    beta_str = "0"
    beta_ax = np.array([0.0])
    alpha_ax = np.array([0.0])
    psi_ax = np.array([0.0])
    BETA3, ALPHA3, PSI3 = np.meshgrid(beta_ax, alpha_ax, psi_ax, indexing='ij')

    # 3. Define other params
    wg_val = 1.0
    
    if len(sys.argv) > 1:
        kernel_path = sys.argv[1]
    else:
        kernel_path = "/data/sguotong/projects/CaGe/data/j_eff_expr/j_eff_kernels.pkl"
    
    print(f"Using kernel file: {kernel_path}", flush=True)

    save_dir = Path("/data/sguotong/projects/CaGe/data/j_eff_test")
    print(f"Output will be saved to: {save_dir.resolve()}", flush=True)


    # 4. Run the computation
    try:
        Jr_plus, Jphi_plus, Jz_plus, Jr_cross, Jphi_cross, Jz_cross = compute_j_eff_6d(
            # beta_str,
            R3, PHI3, Z3,
            BETA3, ALPHA3, PSI3,
            wg_val,
            kernel_path,
            save_dir=save_dir
        )
        
        print(f"\nSuccess! Output shapes:")
        print(f"Jr:   {Jr_plus.shape}")
        print(f"Jphi: {Jphi_plus.shape}")
        print(f"Jz:   {Jz_plus.shape}")

    except FileNotFoundError:
        print(f"Error: Kernel file not found at {kernel_path}", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)