import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from include.custom_types import *

def plot_disk_slice_6d(
    Fr: NDArray,
    Fphi: NDArray,
    Fz: NDArray,
    r: NDArray,           # (Nr,)
    phi: NDArray,         # (Nphi,), usually endpoint=False on [0, 2π)
    z: NDArray,           # (Nz,)
    beta: NDArray,        # (Nb,)
    alpha: NDArray,       # (Na,)
    psi: NDArray,         # (Npsi,)
    *,
    z_at: float | None = None,
    beta_at: float = 0.0,
    alpha_at: float = 0.0,
    psi_at: float = 0.0,
    stream_component: str = "imag",   # "real" or "imag" for stream vectors
    center_zero: bool = False,        # centers colormap for "real"/"imag" magnitude
    vmin: float | None = None,
    vmax: float | None = None,
    shading: str = "auto",
    cmap: str | None = None,
    axes_equal: bool = True,
    add_contours: bool = False,
    n_levels: int = 12,
    # Cartesian resampling grid (streamplot requires rectilinear x,y)
    cart_Nx: int = 300,
    cart_Ny: int = 300,
    interp_kind: str = "cubic",      # "linear" or "nearest"
    stream_density: float | tuple = 1.2,
    stream_linewidth: float = 1.0,
    stream_arrowsize: float = 1.0,
    title: str | None = None,
):
    """
    Plot |F| as a heatmap and overlay a normalized streamplot on a z=const slice.
    Internally resamples the (Fr, Fphi) polar data onto a rectilinear Cartesian grid.

    Shapes: Fr, Fphi, Fz are (Nr, Nphi, Nz, Nb, Na, Npsi).
    Exactly one of (z_at, z_index) must be given.
    """

    z_index = int(np.argmin(np.abs(z - z_at)))
    k = z_index
    beta_index = int(np.argmin(np.abs(beta - beta_at)))
    alpha_index = int(np.argmin(np.abs(alpha - alpha_at)))
    psi_index = int(np.argmin(np.abs(psi - psi_at)))
    print(f"Using indices: z[{k}] beta[{beta_index}] alpha[{alpha_index}] psi[{psi_index}]")

    # ---- extract (r,phi) slices for chosen (z, beta, alpha, psi)
    Fr2   = Fr[:, :, k, beta_index, alpha_index, psi_index]
    Fphi2 = Fphi[:, :, k, beta_index, alpha_index, psi_index]
    Fz2   = Fz[:,  :, k, beta_index, alpha_index, psi_index]
    F_mag = np.sqrt(np.abs(Fr2)**2 + np.abs(Fphi2)**2 + np.abs(Fz2)**2)

    # ---- convert in-plane vector to Cartesian at polar nodes
    Rg, PHIg = np.meshgrid(r, phi, indexing="ij")
    cph, sph = np.cos(PHIg), np.sin(PHIg)
    Fx = Fr2 * cph - Fphi2 * sph
    Fy = Fr2 * sph + Fphi2 * cph

    # choose component for streamplot (real/imag)
    if stream_component == "real":
        U = np.real(Fx)
        V = np.real(Fy)
    elif stream_component == "imag":
        U = np.imag(Fx)
        V = np.imag(Fy)
    else:
        raise ValueError("stream_component must be 'real' or 'imag'.")

    # ---- build periodic extension in phi for interpolation robustness
    # Append a φ=2π column equal to φ=0 column
    phi_ext = np.concatenate([phi, [phi[0] + 2*np.pi]])
    def _extend_phi(A):  # A: (Nr, Nphi)
        return np.concatenate([A, A[:, :1]], axis=1)

    Vmag_ext = _extend_phi(F_mag)
    U_ext    = _extend_phi(U)
    V_ext    = _extend_phi(V)

    # Interpolators on (r, φ) grid
    mag_interp = RegularGridInterpolator(
        (r, phi_ext), Vmag_ext, method=interp_kind,
        bounds_error=False, fill_value=np.nan
    )
    U_interp = RegularGridInterpolator(
        (r, phi_ext), U_ext, method=interp_kind,
        bounds_error=False, fill_value=np.nan
    )
    V_interp = RegularGridInterpolator(
        (r, phi_ext), V_ext, method=interp_kind,
        bounds_error=False, fill_value=np.nan
    )

    # ---- target Cartesian grid (rectilinear, as streamplot expects)
    a = float(np.max(r))
    x = np.linspace(-a, a, cart_Nx)
    y = np.linspace(-a, a, cart_Ny)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")  # shape (Ny, Nx)

    Rc   = np.hypot(Xc, Yc)
    PHIc = np.mod(np.arctan2(Yc, Xc), 2*np.pi)

    # mask outside disk
    disk_mask = Rc <= a

    # evaluate interpolants at Cartesian nodes (Ny, Nx, 2) points
    pts = np.stack([Rc, PHIc], axis=-1)
    Vmag_cart = mag_interp(pts)
    U_cart    = U_interp(pts)
    V_cart    = V_interp(pts)

    # apply disk mask
    Vmag_cart = np.where(disk_mask, Vmag_cart, np.nan)
    U_cart    = np.where(disk_mask, U_cart,    0.0)
    V_cart    = np.where(disk_mask, V_cart,    0.0)

    # ---- normalize stream vectors (unit arrows)
    speed = np.hypot(U_cart, V_cart)
    eps = np.finfo(float).eps
    U_unit = np.where(speed > eps, U_cart / speed, 0.0)
    V_unit = np.where(speed > eps, V_cart / speed, 0.0)

    # ---- plotting (use rectilinear grid for both layers so they align perfectly)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))

    # Use pcolormesh with the rectilinear grid edges; simplest is imshow:
    # but to keep coordinates exact, use pcolormesh on node grid:
    hm = ax.pcolormesh(
        Xc, Yc, Vmag_cart,
        shading=shading, cmap=cmap, vmin=vmin, vmax=vmax    # type: ignore
    )
    cb = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)

    if add_contours:
        cs = ax.contour(Xc, Yc, Vmag_cart, levels=n_levels, linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")

    # streamplot with rectilinear 1D x,y
    ax.streamplot(
        x, y, U_unit, V_unit,
        density=stream_density,
        linewidth=stream_linewidth,
        arrowsize=stream_arrowsize,
        color="k", minlength=0.05, maxlength=0.1,
    )

    # cosmetics
    if axes_equal:
        ax.set_aspect("equal", adjustable="box")
    pad = 0.02 * a
    ax.set_xlim(-a - pad, a + pad)
    ax.set_ylim(-a - pad, a + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # cavity rim
    th = np.linspace(0, 2*np.pi, 512, endpoint=True)
    ax.plot(a*np.cos(th), a*np.sin(th), lw=1.0, alpha=0.8, color="k")

    if title is None:
        z_lbl = f"z≈{z[k]:.5g}"
        meta = f"β[{beta_index}] α[{alpha_index}] ψ[{psi_index}]"
        ax.set_title(f"Disk slice | {z_lbl} | {meta}, stream={stream_component}")
    else:
        ax.set_title(title)

    # optional symmetric limits for signed magnitude maps
    if center_zero and (vmin is None or vmax is None):
        vmax_auto = np.nanmax(np.abs(Vmag_cart))
        ax.collections[0].set_clim(-vmax_auto, +vmax_auto)

    return fig, ax


if __name__ == "__main__":
    from pathlib import Path
    from include.custom_types import *

    # simple test
    Nr, Nphi, Nz = 30, 90, 50
    Nbeta, Nalpha, Npsi = 181, 1, 1
    a, L = 0.0206, 0.2032
    r = np.linspace(0.0, a, Nr)
    phi = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
    z = np.linspace(-L/2, L/2, Nz)
    beta = np.linspace(0.0, np.pi, Nbeta)
    alpha = np.linspace(0.0, 2*np.pi, Nalpha, endpoint=False)
    psi = np.linspace(0.0, 2*np.pi, Npsi, endpoint=False)

    Rg, PHIg, Zg, BETAg, ALPHAg, PSIg = np.meshgrid(
        r, phi, z, beta, alpha, psi,
        indexing="ij"
    )

    # data_dir = Path("/data/sguotong/projects/CaGe/data/j_eff_arr")
    # jr_p = np.load(data_dir / f"Jr_p_{Nr},{Nphi},{Nz},{Nbeta},{Nalpha},{Npsi}.npy")
    # jphi_p = np.load(data_dir / f"Jphi_p_{Nr},{Nphi},{Nz},{Nbeta},{Nalpha},{Npsi}.npy")
    # jz_p = np.load(data_dir / f"Jz_p_{Nr},{Nphi},{Nz},{Nbeta},{Nalpha},{Npsi}.npy")

    from include.cavity_modes import *
    from include.bessel import initialize_bessel_table, BesselJZeros, BesselJpZeros
    initialize_bessel_table(10, 10)
    fam, m, n, p, sign = "TE", 2, 1, 2, "-"
    Er_p, Ephi_p, Ez_p = E_mnp(a, L, Rg, PHIg, Zg, fam, m, n, p, sign)
    Er, Ephi, Ez = np.imag(Er_p), np.imag(Ephi_p), np.imag(Ez_p)

    fig, ax = plot_disk_slice_6d(
        Er, Ephi, Ez,
        r, phi, z,
        beta, alpha, psi,
        z_at=0.01*L,
        beta_at=0.0,
        alpha_at=0.0,
        psi_at=0.0,
        stream_component="imag",
        center_zero=True,
        add_contours=True,
        n_levels=8,
    )
    plt.savefig(f"{fam}{m}{n}{p}{sign}.png", dpi=300)