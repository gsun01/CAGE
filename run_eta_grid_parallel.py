# run_eta_simple.py
from __future__ import annotations
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Your modules
from cavity_modes import CavityModeHelper       # expects CavityModeHelper(a, L).mode_functions(...)
from antenna_pattern import antenna_pattern     # computes eta(beta, alpha, psi)

def _compute_one(args):
    (fam, sign, pol, m, n, p,
     a, L, r, phi, z, beta, alpha, psi,
     outdir, antenna_kwargs) = args

    # Build 3D mesh once per mode (cheap vs field eval; avoids shipping big arrays between processes)
    R3, PHI3, Z3 = np.meshgrid(r, phi, z, indexing="ij")

    # Make mode functions once and (optionally) prewarm caches by evaluating on (R3,PHI3,Z3)
    cav = CavityModeHelper(a, L)
    Er_fn, Ephi_fn, Ez_fn = cav.mode_functions(fam, m, n, p, sign)
    # Precompute once to fill any lru_cache / jitted paths
    _ = Er_fn(R3, PHI3, Z3); _ = Ephi_fn(R3, PHI3, Z3); _ = Ez_fn(R3, PHI3, Z3)

    # Compute eta on the 1D angle grids
    kw = dict(antenna_kwargs or {})
    try:
        # Preferred: if antenna_pattern supports passing field callables explicitly
        eta = antenna_pattern(
            fam, m, n, p, sign, pol, a, L,
            r=r, phi=phi, z=z,
            beta=beta, alpha=alpha, psi=psi,
            Er=Er_fn, Ephi=Ephi_fn, Ez=Ez_fn,
            **kw
        )
    except TypeError:
        # Fallback: older signature without explicit field callables
        eta = antenna_pattern(
            fam, m, n, p, sign, pol, a, L,
            r=r, phi=phi, z=z,
            beta=beta, alpha=alpha, psi=psi,
            **kw
        )

    eta = np.asarray(eta)  # expected shape (Nb, Na, Npsi)
    fname = f"eta_{fam}{sign}_pol{pol}_m{m}_n{n}_p{p}.npy"
    fpath = os.path.join(outdir, fname)
    np.save(fpath, eta)
    return fpath, eta.shape

def run_eta_sweep(
    r: np.ndarray, phi: np.ndarray, z: np.ndarray,
    beta: np.ndarray, alpha: np.ndarray, psi: np.ndarray,
    *, a: float, L: float,
    m_vals, n_vals, p_vals,
    families=("TE", "TM"), signs=("+", "-"), pols=("+", "x"),
    outdir="eta_out", max_workers=None, antenna_kwargs=None
):
    os.makedirs(outdir, exist_ok=True)
    # Save axes once (so you don’t duplicate in every file)
    np.savez(os.path.join(outdir, "axes.npz"),
             r=r, phi=phi, z=z, beta=beta, alpha=alpha, psi=psi)

    combos = list(product(families, signs, pols, m_vals, n_vals, p_vals))
    jobs = [
        (fam, sign, pol, m, n, p,
         a, L, r, phi, z, beta, alpha, psi,
         outdir, antenna_kwargs)
        for (fam, sign, pol, m, n, p) in combos
    ]

    if max_workers is None:
        # Reasonable default; processes avoid GIL for NumPy/SciPy-heavy work
        import os as _os
        max_workers = max(1, (_os.cpu_count() or 1) - 0)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_compute_one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            fpath, shape = fut.result()
            print(f"[{k}/{len(futs)}] saved {os.path.basename(fpath)}  shape={shape}")
            results.append((fpath, shape))
    return results

# --- Example usage ---
if __name__ == "__main__":
    # Example axes (replace with your actual arrays)
    a, L = 0.15, 0.50
    r    = np.linspace(0.0, a, 200, endpoint=True)
    phi  = np.linspace(0.0, 2*np.pi, 256, endpoint=False)
    z    = np.linspace(-L/2, +L/2, 96, endpoint=True)

    beta  = np.linspace(0.0, np.pi, 60)      # Nb
    alpha = np.array([0.0])                  # Na=1
    psi   = np.array([0.0])                  # Npsi=1

    # Mode sweep
    m_vals = [0, 1, 2]
    n_vals = [1, 2]
    p_vals = [0, 1, 2]

    run_eta_sweep(
        r, phi, z, beta, alpha, psi,
        a=a, L=L,
        m_vals=m_vals, n_vals=n_vals, p_vals=p_vals,
        families=("TE", "TM"), signs=("+", "-"), pols=("+", "x"),
        outdir="eta_out",
        antenna_kwargs=dict(normalize=True)  # or {}
    )
