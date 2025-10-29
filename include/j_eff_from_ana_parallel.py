"""
Routine for computing 6D effective current (j_eff) grids
from pickled SymPy expressions using spatial-grid chunking 
for CPU cache efficiency and process-level parallelism to 
compute components concurrently.
"""

from __future__ import annotations
__all__ = ["compute_j_eff_6d"]
import numpy as np
from custom_types import *
import pickle
import sympy as sp
import concurrent.futures
import os
from pathlib import Path

# --- F_k(q) Functions ---
def F1_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    result = -1j/(2*q_safe) - np.exp(-1j*q_safe)/q_safe**2 - 1j*(1-np.exp(-1j*q_safe))/q_safe**3
    return np.where(q == 0, 0.0 + 0.0j, result)

def F2_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    result = -(1+np.exp(-1j*q_safe))/q_safe**2 - 2j*(1-np.exp(-1j*q_safe))/q_safe**3
    return np.where(q == 0, 0.0 + 0.0j, result)

def F2p_func(q):
    q_safe = np.where(q == 0, 1e-18, q)
    exp_term = np.exp(-1j*q_safe)
    one_minus_exp = (1 - exp_term)
    result = (1j*exp_term/q_safe**2 
            - 2*(-1 - exp_term)/q_safe**3 
            + 2.0*exp_term/q_safe**3 
            + 6.0*1j*one_minus_exp/q_safe**4)
    return np.where(q == 0, 0.0 + 0.0j, result)

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
    return np.stack([[exx_p, exy_p, exz_p, eyy_p, eyz_p, ezz_p], [exx_c, exy_c, exz_c, eyy_c, eyz_c, ezz_c]], axis=0).astype(c128)

def n_vec_6d(BETA3: NDArray, ALPHA3: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    sb, cb = np.sin(BETA3), np.cos(BETA3)
    ca, sa = np.cos(ALPHA3), np.sin(ALPHA3)
    nx = sb*ca; ny = sb*sa; nz = cb
    return nx, ny, nz

def q_field_6d(R3: NDArray, PHI3: NDArray, Z3: NDArray, 
               BETA3: NDArray, ALPHA3: NDArray, wg: float) -> NDArray:
    R6   = R3[..., None, None, None]; PHI6 = PHI3[..., None, None, None]
    Z6   = Z3[..., None, None, None]; ALPHA6 = ALPHA3[None, None, None, ...]
    BETA6  = BETA3[None, None, None, ...]; sb, cb = np.sin(BETA6), np.cos(BETA6)
    return wg * ( R6 * sb * np.cos(PHI6 - ALPHA6) + Z6 * cb )

# --- Core Routines ---

def load_symbolic_kernels(path: str) -> tuple[tuple[dict, ...], tuple]:
    """
    Loads pickled SymPy expressions and returns the symbolic maps
    and the argument symbols.
    """
    print(f"Loading symbolic kernels from: {path}", flush=True)
    
    with open(path, "rb") as f:
        pack = pickle.load(f)
        
    var_names = pack["vars"]
    arg_symbols = sp.symbols(' '.join(var_names))

    kernels_sym = (
        pack["JR_plus_exprs"],
        pack["JPHI_plus_exprs"],
        pack["JZ_plus_exprs"],
        pack["JR_cross_exprs"],
        pack["JPHI_cross_exprs"],
        pack["JZ_cross_exprs"]
    )
    
    print(f"Symbolic kernels loaded. Args: {var_names}", flush=True)
    return kernels_sym, arg_symbols

def compile_lambdify_map(symbolic_map: dict, arg_symbols: tuple) -> dict:
    """
    Compiles a map of symbolic expressions into lambdified functions.
    """
    numpy_modules = {
        'numpy': np,
        'I': 1j,
        'pi': np.pi
    }
    
    return {k: sp.lambdify(arg_symbols, v, modules=[numpy_modules], cse=True)
            for k, v in symbolic_map.items()}

def precalculate_fk_map(q: NDArray) -> dict:
    """
    Calculates the Fk map once from the 6D q-field.
    """
    print("    Pre-calculating Fk values...", flush=True)
    Fk_slice_map = {}
    q_slice_sq = q * q
    F1_s = F1_func(q); Fk_slice_map[0] = F1_s; Fk_slice_map[1] = q * F1_s; del F1_s
    F2_s = F2_func(q); Fk_slice_map[2] = F2_s; Fk_slice_map[3] = q * F2_s; del F2_s
    F2p_s = F2p_func(q); Fk_slice_map[4] = F2p_s; Fk_slice_map[5] = q * F2p_s; Fk_slice_map[6] = q_slice_sq * F2p_s; del F2p_s, q_slice_sq
    print("    ...Fk pre-calculation complete.", flush=True)
    return Fk_slice_map

def eval_component_grid_6d(
    coeff_map: dict, 
    coeff_map_sym: dict, 
    e6: NDArray, 
    R3: NDArray, PHI3: NDArray, Z3: NDArray, 
    nx: NDArray, ny: NDArray, nz: NDArray, 
    q: NDArray,
    wg_val: float,
    Fk_slice_map: dict
) -> NDArray:
    """
    Evaluates one component (Jr, Jphi, or Jz) on the full 6D grid
    by iterating over the spatial grid.
    """
    Nr, Nphi, Nz = R3.shape
    Nbeta, Nalpha, Npsi = nx.shape
    acc = np.zeros((Nr, Nphi, Nz, Nbeta, Nalpha, Npsi), dtype=c128)
    
    total_spatial_points = Nr * Nphi * Nz
    print(f"  Starting component evaluation ({len(coeff_map)} terms) over {total_spatial_points} spatial points...", flush=True)

    for i in range(Nr):
        for j in range(Nphi):
            for k in range(Nz):
                r_val = R3[i, j, k]
                phi_val = PHI3[i, j, k]
                z_val = Z3[i, j, k]
                q_slice = q[i, j, k, :, :, :]
                
                acc_slice = np.zeros((Nbeta, Nalpha, Npsi), dtype=c128)
                
                for (term_j, term_k), func in coeff_map.items():
                    
                    try:
                        C_slice = func(r_val, phi_val, z_val, nx, ny, nz, q_slice, wg_val)
                        
                        Fk_slice = Fk_slice_map[term_k][i, j, k, :, :, :]
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

    print(f"  ... component evaluation finished for all {total_spatial_points} points.", flush=True)
    return acc

# --- Parallelism Helpers ---

def init_worker():
    """Forces numpy's LinAlg backends to be single-threaded in worker processes."""
    print(f"Initializing worker {os.getpid()}... setting threads to 1.", flush=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

def eval_component_grid_6d_wrapper(args):
    """
    Simple wrapper to unpack args for ProcessPoolExecutor.submit
    and return the result with its name.
    
    This function runs inside the worker process.
    """
    (coeff_map_sym, arg_symbols, e6, R3, PHI3, Z3, 
     nx, ny, nz, q, wg_val, Fk_slice_map, name) = args
    
    # Compile the functions inside each worker
    print(f"--- [Worker {os.getpid()}] Compiling component: {name} ---", flush=True)
    coeff_map = compile_lambdify_map(coeff_map_sym, arg_symbols)
    
    print(f"--- [Worker {os.getpid()}] Starting evaluation: {name} ---", flush=True)
    result = eval_component_grid_6d(
        coeff_map, coeff_map_sym, e6, R3, PHI3, Z3, 
        nx, ny, nz, q, wg_val, Fk_slice_map
    )
    
    print(f"--- [Worker {os.getpid()}] Finished component: {name} ---", flush=True)
    return (name, result)

# --- Main Entry Point ---

def compute_j_eff_6d(
    path: str,
    R3: NDArray, 
    PHI3: NDArray, 
    Z3: NDArray,
    BETA3: NDArray, 
    ALPHA3: NDArray, 
    PSI3: NDArray,
    wg: float,
    save_dir: str | os.PathLike | None = None
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    
    # compiled functions cannot be passed to workers
    # only load symbolic expressions and pass those to each worker process to be compiled
    kernels_sym, arg_symbols = load_symbolic_kernels(path)
    JR_plus_sym, JPHI_plus_sym, JZ_plus_sym, JR_cross_sym, JPHI_cross_sym, JZ_cross_sym = kernels_sym

    print("Computing n-vectors, pol tensors, and q-field...", flush=True)
    nx, ny, nz = n_vec_6d(BETA3, ALPHA3)
    e6 = pol_tensor_6d(BETA3, ALPHA3, PSI3)
    e6_p, e6_c = e6[0], e6[1]
    q = q_field_6d(R3, PHI3, Z3, BETA3, ALPHA3, wg)
    
    Fk_slice_map = precalculate_fk_map(q)

    tasks = [
        (JR_plus_sym, arg_symbols, e6_p, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jr_p"),
        (JPHI_plus_sym, arg_symbols, e6_p, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jphi_p"),
        (JZ_plus_sym, arg_symbols, e6_p, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jz_p"),
        (JR_cross_sym, arg_symbols, e6_c, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jr_c"),
        (JPHI_cross_sym, arg_symbols, e6_c, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jphi_c"),
        (JZ_cross_sym, arg_symbols, e6_c, R3, PHI3, Z3, nx, ny, nz, q, wg, Fk_slice_map, "Jz_c")
    ]

    results = {}
    print("--- Starting j_eff component computation in parallel (6 processes) ---", flush=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, initializer=init_worker) as executor:
        
        future_to_name = {executor.submit(eval_component_grid_6d_wrapper, task): task[-1] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                name, result_array = future.result()
                results[name] = result_array
            except Exception as exc:
                print(f'{name} generated an exception: {exc}', flush=True)
                raise exc

    print("--- j_eff computation complete. ---", flush=True)

    # save results to disk
    if save_dir is not None:
        data_dir = Path(save_dir)
        try:
            data_dir.mkdir(parents=True, exist_ok=True)            
            for name, array in results.items():
                file_path = data_dir / f"{name}_{Nr},{Nphi},{Nz},{Nbeta},{Nalpha},{Npsi}.npy"
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
    
    # 1. Define spatial grid (e.g., 30x20x40)
    Nr, Nphi, Nz = 30, 20, 10
    r_ax = np.linspace(0, 1, Nr)
    phi_ax = np.linspace(0, 2*np.pi, Nphi)
    z_ax = np.linspace(-1, 1, Nz)
    R3, PHI3, Z3 = np.meshgrid(r_ax, phi_ax, z_ax, indexing='ij')

    # 2. Define angular grid (e.g., 10x10x5)
    Nbeta, Nalpha, Npsi = 5, 4, 3
    beta_ax = np.linspace(0, np.pi, Nbeta)
    alpha_ax = np.linspace(0, 2*np.pi, Nalpha)
    psi_ax = np.linspace(0, 2*np.pi, Npsi)
    BETA3, ALPHA3, PSI3 = np.meshgrid(beta_ax, alpha_ax, psi_ax, indexing='ij')

    # 3. Define other params
    wg_val = 1.0
    
    if len(sys.argv) > 1:
        kernel_path = sys.argv[1]
    else:
        kernel_path = "/grad/sguotong/projects/CaGe/data/j_eff_expr/j_eff_kernels.pkl"
    
    print(f"Using kernel file: {kernel_path}", flush=True)

    save_dir = Path("/grad/sguotong/projects/CaGe/data/j_eff_arr")
    print(f"Output will be saved to: {save_dir.resolve()}", flush=True)


    # 4. Run the computation
    try:
        Jr_plus, Jphi_plus, Jz_plus, Jr_cross, Jphi_cross, Jz_cross = compute_j_eff_6d(
            kernel_path,
            R3, PHI3, Z3,
            BETA3, ALPHA3, PSI3,
            wg=wg_val,
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

