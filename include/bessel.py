from __future__ import annotations
__all__ = [
    "initialize_bessel_table",
    "BesselJZeros", "BesselJpZeros",
    "get_bessel_tables", "is_initialized",
]

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

# Module-level caches
_J_TABLE: np.ndarray | None = None   # shape: (m_max+1, n_max)
_JP_TABLE: np.ndarray | None = None  # shape: (m_max+1, n_max)

def is_initialized() -> bool:
    """Return True iff the tables are already built."""
    return _J_TABLE is not None and _JP_TABLE is not None

def initialize_bessel_table(m_max: int, n_max: int, *, overwrite: bool = False) -> tuple[int, int]:
    """
    Precompute and cache zeros of J_m and J'_m.
    After this is called once (unless overwrite=True), lookups are O(1).

    Parameters
    ----------
    m_max : int
        Maximum Bessel order m to precompute (inclusive). Must be >= 0.
    n_max : int
        Number of zeros per order to precompute (>= 1).
    overwrite : bool
        If True, rebuilds the tables even if they already exist.

    Returns
    -------
    (m_size, n_size) : tuple[int, int]
        The shape of the precomputed tables: (m_max+1, n_max).
    """
    global _J_TABLE, _JP_TABLE

    if m_max < 0 or n_max <= 0:
        raise ValueError("m_max must be >= 0 and n_max must be >= 1.")

    if is_initialized() and not overwrite:
        return _J_TABLE.shape  # type: ignore[union-attr]

    # Build fresh tables
    J = np.empty((m_max + 1, n_max), dtype=float)
    JP = np.empty((m_max + 1, n_max), dtype=float)

    for m in range(m_max + 1):
        # the first n_max positive zeros, sorted
        J[m, :]  = jn_zeros(m,  n_max)
        JP[m, :] = jnp_zeros(m, n_max)

    _J_TABLE, _JP_TABLE = J, JP
    return J.shape  # type: ignore[union-attr]

def get_bessel_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return references to the cached tables (read-only use recommended)."""
    if not is_initialized():
        raise RuntimeError("Bessel tables are not initialized. Call initialize_bessel_table(...) first.")
    return _J_TABLE, _JP_TABLE  # type: ignore[return-value]

def _check_bounds(m: int, n: int) -> None:
    if not is_initialized():
        raise RuntimeError("Bessel tables are not initialized. Call initialize_bessel_table(...) first.")
    if m < 0 or n <= 0:
        raise IndexError("Use m >= 0 and n >= 1 (n is the 1-based zero index).")
    m_size, n_size = _J_TABLE.shape  # type: ignore[union-attr]
    if m >= m_size or n > n_size:
        raise IndexError(
            f"Requested (m={m}, n={n}) is outside precomputed range "
            f"(m <= {m_size-1}, n <= {n_size}). Re-run initialize_bessel_table with larger limits."
        )

def BesselJZeros(m: int, n: int) -> float:
    """
    Return the n-th zero of J_m(x).
    Convention: m is 0-based order; n is 1-based zero index (n >= 1).
    """
    _check_bounds(m, n)
    return _J_TABLE[m, n - 1]  # type: ignore[index]

def BesselJpZeros(m: int, n: int) -> float:
    """
    Return the n-th zero of J'_m(x).
    Convention: m is 0-based order; n is 1-based zero index (n >= 1).
    """
    _check_bounds(m, n)
    return _JP_TABLE[m, n - 1]  # type: ignore[index]