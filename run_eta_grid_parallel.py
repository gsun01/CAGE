from __future__ import annotations
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import traceback

from include.custom_types import *
from include.cavity_modes import *
from include.antenna_pattern import antenna_pattern
from include.bessel import initialize_bessel_table

f64 = np.float64 | float
c128 = np.complex128 | float

def _init_worker():
    initialize_bessel_table(20, 20)

def _compute_one_mnp(args):
    try:
        (fam, m, n, p,
        a, L, r, phi, z, beta, alpha, psi,
        kernel_path, eta_save_dir, j_eff_save_dir) = args

        R3, PHI3, Z3 = np.meshgrid(r, phi, z, indexing="ij")
        BETA3, ALPHA3, PSI3 = np.meshgrid(beta, alpha, psi, indexing="ij")

        # compute cavity mode for fam,m,n,p
        if fam == "TE" and (m == 0 or p == 0):
            # print(f"Skipping unphysical TE_0n0 mode", flush=True)
            return ("ok", "skipped", None)

        Er3, Ephi3, Ez3 = E_mnp(a, L, R3, PHI3, Z3, fam, m, n, p)
        w_mnp = omega_mnp(a, L, fam, m, n, p)
        print(f"Computed cavity mode {fam}_{m}{n}{p} at ω={w_mnp:.3f}.", flush=True)
        # do E^2 integral on (R3,PHI3,Z3) grid to get a scalar norm
        Vcav = np.pi * a**2 * L
        E2 = np.abs(Er3)**2 + np.abs(Ephi3)**2 + np.abs(Ez3)**2
        # print(f"Is E field finite? {np.isfinite(E2).all()}", flush=True)
        if not np.isfinite(E2).all():
            raise ValueError(f"Non-finite fields at {fam}{m}{n}{p}.")
        if np.max(E2) == 0.0:
            print(f"All-zero fields at {fam}{m}{n}{p}.", flush=True)
        norm = np.sqrt(Vcav * np.trapezoid(np.trapezoid(np.trapezoid(E2*R3, z, axis=2), phi, axis=1), r, axis=0))
        print(f"Norm is {norm} for {fam}{m}{n}{p}. Computing eta...", flush=True)


        eta_p_save_path = eta_save_dir / f"eta_{fam}_{m}{n}{p}_plus.npy"
        eta_c_save_path = eta_save_dir / f"eta_{fam}_{m}{n}{p}_cross.npy"

        eta_p, eta_c = antenna_pattern(
            R3, PHI3, Z3,
            BETA3, ALPHA3, PSI3,
            Er3, Ephi3, Ez3, norm, w_mnp,
            kernel_path,
            eta_p_save_path,
            eta_c_save_path,
            j_eff_save_dir=j_eff_save_dir
        )

        return ("ok", str(eta_save_dir), tuple(eta_p.shape))
    except Exception as e:
        print(f"Error in worker for args {args}: {e}", flush=True)
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return ("err", tb, None)

def run_eta_sweep(
    r: NDArray, phi: NDArray, z: NDArray,
    beta: NDArray, alpha: NDArray, psi: NDArray,
    a: f64, L: f64, m_vals, n_vals, p_vals,
    kernel_path: str | os.PathLike,
    eta_save_dir: str | os.PathLike, 
    j_eff_save_dir: str | os.PathLike, 
    max_workers=8
    ):


    families = ["TE", "TM"]
    combos = list(product(families, m_vals, n_vals, p_vals))
    jobs = [
        (fam, m, n, p,
         a, L, r, phi, z, beta, alpha, psi,
         kernel_path, eta_save_dir, j_eff_save_dir)
        for (fam, m, n, p) in combos
    ]

    print(f"Jobs to do: {len(jobs)}", flush=True)

    if max_workers is None:
        import os as _os
        max_workers = max(1, (_os.cpu_count() or 1) - 0)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=()) as ex:
        futs = [ex.submit(_compute_one_mnp, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            print(f"Working on job {k}/{len(futs)}...", flush=True)
            status, payload, shape = fut.result()
            if status == "ok":
                print(f"[{k}/{len(futs)}] saved {Path(payload).name}  shape={shape}", flush=True)
            else:
                print(f"[{k}/{len(futs)}] WORKER ERROR:\n{payload}", flush=True)
    
    print("All done!", flush=True)
    return results

if __name__ == "__main__":

#######################################################################
######################### PARAMETERS ##################################
#######################################################################

    # cavity dimensions
    a, L = 0.0206, 0.2032

    # grid sizes
    Nr, Nphi, Nz = 30, 90, 50
    Nbeta, Nalpha, Npsi = 60, 1, 1
    
    # mode sweep parameters
    m_vals = [0, 1, 2]
    n_vals = [1, 2]
    p_vals = [0, 1, 2]

    # output paths
    datadir = Path("/data/sguotong/projects/CaGe/data")
    kernel_path = datadir / "j_eff_expr" / "j_eff_kernels.pkl"
    eta_save_dir = datadir / "etas"
    j_eff_save_dir = datadir / "j_eff_arr"

#######################################################################

    datadir.mkdir(parents=True, exist_ok=True)
    eta_save_dir.mkdir(parents=True, exist_ok=True)
    eta_save_dir.mkdir(parents=True, exist_ok=True)

    _init_worker()

    r    = np.linspace(0.0, a, Nr, endpoint=True)
    phi  = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
    z    = np.linspace(-L/2, +L/2, Nz, endpoint=True)

    beta  = np.linspace(0.0, np.pi, Nbeta, endpoint=True)
    alpha = np.linspace(0.0, 2*np.pi, Nalpha, endpoint=False)
    psi   = np.linspace(0.0, 2*np.pi, Npsi, endpoint=False)

    print(f"Running eta sweep for a={a}, L={L}", flush=True)

    run_eta_sweep(
        r, phi, z,
        beta, alpha, psi,
        a, L, m_vals, n_vals, p_vals,
        kernel_path,
        eta_save_dir,
        j_eff_save_dir,
        max_workers=8
    )


    
