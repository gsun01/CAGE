import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, jn_zeros, jnp_zeros

c = 299_792_458.0
mu0 = 4*np.pi*1e-7
Lcav = 1.00    # m
Rcav = 1.00     # m
# Rcav = 0.21    # m
# Rcav = 0.155    # m

def make_cylinder_grid_uniform(R=Rcav, L=Lcav, Nr=40, Nphi=128, Nz=48):
    dr = R / Nr
    dz = L / Nz
    dphi = 2*np.pi / Nphi

    r = (np.arange(Nr) + 0.5) * dr
    phi = np.arange(Nphi) * dphi
    z = -0.5*L + (np.arange(Nz) + 0.5) * dz

    Rg, Pg, Zg = np.meshgrid(r, phi, z, indexing="ij")

    x = Rg * np.cos(Pg)
    y = Rg * np.sin(Pg)

    W = Rg * dr * dphi * dz

    return x.ravel(), y.ravel(), Zg.ravel(), Rg.ravel(), Pg.ravel(), W.ravel()

def make_cylinder_grid_quad(R=Rcav, L=Lcav, Nr=40, Nphi=128, Nz=48):
    ur, wr = np.polynomial.legendre.leggauss(Nr)
    uz, wz = np.polynomial.legendre.leggauss(Nz)

    r = 0.5 * R * (ur + 1.0)
    wr = 0.5 * R * wr

    z = 0.5 * L * uz          # centered coordinate: [-L/2, L/2]
    wz = 0.5 * L * wz

    phi = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
    wphi = 2*np.pi / Nphi

    Rg, Pg, Zg = np.meshgrid(r, phi, z, indexing="ij")

    x = Rg * np.cos(Pg)
    y = Rg * np.sin(Pg)

    W = wr[:, None, None] * Rg * wphi * wz[None, None, :]

    return x.ravel(), y.ravel(), Zg.ravel(), Rg.ravel(), Pg.ravel(), W.ravel()

############################################
# cavity modes
############################################

def bessel_zero(m: int, n: int, mode='TM'):
    '''
    Returns the n-th root of Jm for TM modes and n-th root of Jm' for
    TE modes.
    '''
    if mode == 'TM': return jn_zeros(m,5)[n-1]
    else: return jnp_zeros(m,5)[n-1]

def E_mnp(r, phi, z, m: int, n: int, p: int, mode='TM', sign='+', R=Rcav, L=Lcav):
    '''
    Returns arrays Ex, Ey, Ez with r.shape for the cavity mode.
    '''
    if m < 0 or p < 0 or n <= 0: 
        print('m, p must be non-negative, and n must be positive.')
        return
    _sign = '-' if (m == 0 and mode == 'TM') else sign

    g = bessel_zero(m, n, mode)/R
    kz = np.pi*p/L
    k0 = np.sqrt(g*g+kz*kz) # k0^2 = g^2 + kz^2 --> k0^2 - kz^2 = g^2
    omega = k0*c

    kz_g, kz_g2 = kz/g, kz/(g*g)
    k_g2 = k0 / (g*g)
    smp = np.sin(m*phi)
    cmp = np.cos(m*phi)
    s_kz_z = np.sin(kz*z)
    c_kz_z = np.cos(kz*z)

    Jm = jv(m, g*r)
    Jm_p = jvp(m, g*r)

    def m_over_r_Jm(m, g, r):
        x = g*r
        if m == 0:
            return np.zeros_like(r, dtype=np.result_type(r, float))
        return 0.5*g*(jv(m-1, x) + jv(m+1, x))
    m_r_Jm = m_over_r_Jm(m, g, r)
    # m_r_Jm = m/r * Jm

    if mode == 'TM':
        Er_TM = -kz_g * (smp if _sign == '+' else cmp) * s_kz_z * Jm_p
        Ephi_TM = -kz_g2 * (cmp if _sign == '+' else -smp) * s_kz_z * m_r_Jm
        Ez_TM = (smp if _sign == '+' else cmp) * c_kz_z * Jm
        Ex_TM = Er_TM*np.cos(phi) - Ephi_TM*np.sin(phi)
        Ey_TM = Er_TM*np.sin(phi) + Ephi_TM*np.cos(phi)
        return Ex_TM, Ey_TM, Ez_TM, omega
    else:   # TE
        Er_TE = 1j * k_g2 * (cmp if _sign == '+' else -smp) * s_kz_z * m_r_Jm
        Ephi_TE = -1j * k0 / g * (smp if _sign == '+' else cmp) * s_kz_z * Jm_p
        Ez_TE = np.zeros_like(Er_TE)
        Ex_TE = Er_TE*np.cos(phi) - Ephi_TE*np.sin(phi)
        Ey_TE = Er_TE*np.sin(phi) + Ephi_TE*np.cos(phi)
        return Ex_TE, Ey_TE, Ez_TE, omega

############################################
# GW strain tensor
############################################

def berlin_correction(q, eps=1e-2):
    '''
    Returns the high-freq correction factors, i.e.
    bracketed factors in Berlin et al (2022) eq 5.

    q = kg*nhat.x, where nhat is the GW propagation direction
    '''
    F00 = np.empty_like(q, dtype=complex)
    F0i = np.empty_like(q, dtype=complex)
    Fij = np.empty_like(q, dtype=complex)

    # spatial   derivatives
    dF00 = np.empty_like(q, dtype=complex)
    dF0i = np.empty_like(q, dtype=complex)
    dFij = np.empty_like(q, dtype=complex)

    small = np.abs(q) < eps
    large = ~small

    ql = q[large]
    exp_q = np.exp(-1j * ql)
    one_minus = 1.0 - exp_q

    F00[large] = -1j / ql + one_minus / ql**2
    F0i[large] = (-1j / (2.0 * ql) - exp_q / ql**2 - 1j * one_minus / ql**3)
    Fij[large] = (-(1.0 + exp_q) / ql**2 - 2j * one_minus / ql**3)
    dF00[large] = (1j / ql**2 + 1j * exp_q / ql**2 - 2.0 * one_minus / ql**3)
    dF0i[large] = (1j / (2.0 * ql**2) + 1j * exp_q / ql**2 + 3.0 * exp_q / ql**3 + 3j * one_minus / ql**4)
    dFij[large] = (1j * exp_q / ql**2 + (2.0 + 4.0 * exp_q) / ql**3 + 6j * one_minus / ql**4)

    qs = q[small]
    # small q approximations to avoid singularity at spatial origin
    F00[small] = (0.5 - 1j * qs / 6.0 - qs**2 / 24.0)
    F0i[small] = (1.0 / 3.0 - 1j * qs / 8.0 - qs**2 / 30.0)
    Fij[small] = (1.0 / 6.0 - 1j * qs / 12.0 - qs**2 / 40.0)
    dF00[small] = (-1j / 6.0 - qs / 12.0 + 1j * qs**2 / 40.0 + qs**3 / 180.0)
    dF0i[small] = (-1j / 8.0 - qs / 15.0 + 1j * qs**2 / 48.0 + qs**3 / 210.0)
    dFij[small] = (-1j / 12.0 - qs / 20.0 + 1j * qs**2 / 60.0 + qs**3 / 252.0)
    return F00, F0i, Fij, dF00, dF0i, dFij

def gw_basis(beta, alpha, psi=0.0):
    n = np.array([np.sin(beta) * np.cos(alpha),
                np.sin(beta) * np.sin(alpha),
                np.cos(beta)])
    
    e1 = np.array([-np.sin(alpha),
                np.cos(alpha),
                0.0])
    
    e2 = np.array([np.cos(beta) * np.cos(alpha),
                np.cos(beta) * np.sin(alpha),
                -np.sin(beta)])

    u = np.cos(psi) * e2 + np.sin(psi) * e1
    v = -np.sin(psi) * e2 + np.cos(psi) * e1

    e_plus = np.outer(u, u) - np.outer(v, v)
    e_cross = np.outer(u, v) + np.outer(v, u)

    return n, e_plus, e_cross

def normalized_hPD_components(x, y, z, omega_g, beta, alpha, psi=0.0):
    '''
    Berlin et al (2022) eq 5.
    Returns normalized components hPD/(omega_g^2 * h_plus) or hPD/(omega_g^2 * h_cross)
    '''
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, z must have the same shape.")
    if x.ndim != 1:
        raise ValueError("This function expects 1D point arrays.")
    
    n, ep, ec = gw_basis(beta, alpha, psi)
    kg = omega_g / c
    pos = np.column_stack([x,y,z])  # shape (N,3)
    ndotx = pos @ n                 # shape (N,)
    q = kg * ndotx
    F00, F0i, Fij, dF00, dF0i, dFij = berlin_correction(q)

    # V_i = hTT_ij x^j
    Vp = np.einsum("ij,pj->pi", ep, pos)      # shape (N, 3)
    Vc = np.einsum("ij,pj->pi", ec, pos)

    # H = hTT_ij x^i x^j = x^i V_i
    Hp = np.einsum("pi,pi->p", pos, Vp)         # shape (N,)
    Hc = np.einsum("pi,pi->p", pos, Vc)

    h00p, h00c = -Hp*F00, -Hc*F00

    # S_i = ndotx V_i - n_i H
    Sp = ndotx[:, None] * Vp - Hp[:, None] * n[None, :]
    Sc = ndotx[:, None] * Vc - Hc[:, None] * n[None, :]
    h0ip = -Sp * F0i[:, None]
    h0ic = -Sc * F0i[:, None]

    # T_ij = ndotx(n_i V_j + n_j V_i) - hTT_ij ndotx^2 - n_i n_j H
    nn = np.outer(n, n)
    Tp = (ndotx[:, None, None] * (n[None, :, None] * Vp[:, None, :] + n[None, None, :] * Vp[:, :, None])
        - ep[None, :, :] * ndotx[:, None, None]**2 - nn[None, :, :] * Hp[:, None, None])
    Tc = (ndotx[:, None, None] * (n[None, :, None] * Vc[:, None, :] + n[None, None, :] * Vc[:, :, None])
        - ec[None, :, :] * ndotx[:, None, None]**2 - nn[None, :, :] * Hc[:, None, None])
    hijp, hijc = Tp * Fij[:, None, None], Tc * Fij[:, None, None]

    # ------------------------------------------------------------
    # Spatial derivatives of normalized hPD components.
    #
    # partial_a ndotx = n_a
    # partial_a q = kg n_a
    # partial_a V_i = hTT_ia
    # partial_a H = 2 V_a
    # ------------------------------------------------------------

    dh00p = -(2.0 * Vp * F00[:, None] + Hp[:, None] * dF00[:, None] * kg * n[None, :])
    dh00c = -(2.0 * Vc * F00[:, None] + Hc[:, None] * dF00[:, None] * kg * n[None, :])
    dSp = (n[None, :, None] * Vp[:, None, :] + ndotx[:, None, None] * ep.T[None, :, :] - 2.0 * Vp[:, :, None] * n[None, None, :])
    dSc = (n[None, :, None] * Vc[:, None, :] + ndotx[:, None, None] * ec.T[None, :, :] - 2.0 * Vc[:, :, None] * n[None, None, :])
    dh0ip = -(dSp * F0i[:, None, None] + Sp[:, None, :] * dF0i[:, None, None] * kg * n[None, :, None])
    dh0ic = -(dSc * F0i[:, None, None] + Sc[:, None, :] * dF0i[:, None, None] * kg * n[None, :, None])
    dTp = (n[None, :, None, None] * (n[None, None, :, None] * Vp[:, None, None, :]+ n[None, None, None, :] * Vp[:, None, :, None])
        + ndotx[:, None, None, None] * (n[None, None, :, None] * ep.T[None, :, None, :]+ n[None, None, None, :] * ep.T[None, :, :, None])
        - 2.0 * ndotx[:, None, None, None] * n[None, :, None, None] * ep[None, None, :, :]
        - 2.0 * Vp[:, :, None, None] * nn[None, None, :, :])
    dTc = (n[None, :, None, None] * (n[None, None, :, None] * Vc[:, None, None, :]+ n[None, None, None, :] * Vc[:, None, :, None])
        + ndotx[:, None, None, None] * (n[None, None, :, None] * ec.T[None, :, None, :]+ n[None, None, None, :] * ec.T[None, :, :, None])
        - 2.0 * ndotx[:, None, None, None] * n[None, :, None, None] * ec[None, None, :, :]
        - 2.0 * Vc[:, :, None, None] * nn[None, None, :, :])
    dhijp = (dTp * Fij[:, None, None, None] + Tp[:, None, :, :] * dFij[:, None, None, None] * kg * n[None, :, None, None])
    dhijc = (dTc * Fij[:, None, None, None] + Tc[:, None, :, :] * dFij[:, None, None, None] * kg * n[None, :, None, None])

    return h00p, h00c, h0ip, h0ic, hijp, hijc, dh00p, dh00c, dh0ip, dh0ic, dhijp, dhijc, n

############################################
# effective current and overlap factor
############################################

def normalized_effective_current(x, y, z, omega_g, beta, alpha, psi=0.0):
    '''
    Returns the six normalized spatial current components:
        Jxp, Jyp, Jzp, Jxc, Jyc, Jzc, n
    where
        Jip = j_i_plus  / (B0 h_plus)
        Jic = j_i_cross / (B0 h_cross)
    The returned currents have dimensions 1/meter.
    '''
    kg = omega_g / c

    (h00p, h00c, h0ip, h0ic, hijp, hijc,
        dh00p, dh00c, dh0ip, dh0ic, dhijp, dhijc, n,
    ) = normalized_hPD_components(x, y, z, omega_g, beta, alpha, psi)

    def one_pol_current(h00, h0i, hij, dh00, dh0i, dhij):
        '''
        Compute j_i / (B0 h_pol) from normalized hbar = hPD/(k_g^2 h_pol).
        '''

        htrace = (-h00 + hij[:, 0, 0] + hij[:, 1, 1] + hij[:, 2, 2])
        dhtrace = (-dh00 + dhij[:, :, 0, 0] + dhij[:, :, 1, 1] + dhij[:, :, 2, 2])
        # dhtrace[:, a] = partial_a htrace_bar

        # Convert normalized spatial derivatives to derivatives of actual h / h_pol:
        #
        # h_actual / h_pol = k_g^2 hbar
        #
        # partial_a(h_actual / h_pol) = k_g^2 partial_a hbar
        dhtrace_phys = kg**2 * dhtrace
        dh0i_phys = kg**2 * dh0i
        dhij_phys = kg**2 * dhij

        d0h0i_phys = 1j * kg**3 * h0i

        Jx = (
            0.5 * dhtrace_phys[:, 1]
            + d0h0i_phys[:, 1]
            - dhij_phys[:, 1, 1, 1]
            - dhij_phys[:, 2, 2, 1]
            - dhij_phys[:, 1, 0, 0]
        )

        Jy = (
            -0.5 * dhtrace_phys[:, 0]
            - d0h0i_phys[:, 0]
            + dhij_phys[:, 0, 0, 0]
            + dhij_phys[:, 2, 2, 0]
            + dhij_phys[:, 0, 1, 1]
        )

        Jz = (
            dhij_phys[:, 0, 2, 1]
            - dhij_phys[:, 1, 2, 0]
        )

        return Jx, Jy, Jz

    Jxp, Jyp, Jzp = one_pol_current(
        h00p, h0ip, hijp,
        dh00p, dh0ip, dhijp,
    )

    Jxc, Jyc, Jzc = one_pol_current(
        h00c, h0ic, hijc,
        dh00c, dh0ic, dhijc,
    )

    return Jxp, Jyp, Jzp, Jxc, Jyc, Jzc, n

def form_factors(mode, m, n, p, 
                R=Rcav, L=Lcav, grid=None):
    '''
    Computes the overlap factor eta, as defined in
    Berlin et al. (2022), eq. 22.
    '''
    if grid is None:
        # grid = make_cylinder_grid_quad(R=R, L=L)
        grid = make_cylinder_grid_uniform(R=R, L=L)
    x, y, z, r, phi, weights = grid
    z_cap = z + 0.5*L            # Strangely, Berlin's cavity modes are defined with the origin at the bottom cap, not the center of the cylinder
    Ex_plus, Ey_plus, Ez_plus, omega_g = E_mnp(r, phi, z_cap, m, n, p, mode)
    Ex_minus, Ey_minus, Ez_minus, omega_g = E_mnp(r, phi, z_cap, m, n, p, mode, sign='-')

    betas = np.linspace(0, 2*np.pi, 90)
    alpha, psi = np.pi/2.0, 0.0
    etaps, etacs = np.array([]), np.array([])

    denom_plus = np.sum(weights) * np.sum(weights * (np.abs(Ex_plus)**2 + np.abs(Ey_plus)**2 + np.abs(Ez_plus)**2))
    denom_minus = np.sum(weights) * np.sum(weights * (np.abs(Ex_minus)**2 + np.abs(Ey_minus)**2 + np.abs(Ez_minus)**2))

    for beta in betas:
        Jxp, Jyp, Jzp, Jxc, Jyc, Jzc, nn = normalized_effective_current(x, y, z, omega_g, beta, alpha)
        overlap_p_plus = np.sum(weights * (np.conjugate(Ex_plus) * Jxp + np.conjugate(Ey_plus) * Jyp + np.conjugate(Ez_plus) * Jzp))
        overlap_c_plus = np.sum(weights * (np.conjugate(Ex_plus) * Jxc + np.conjugate(Ey_plus) * Jyc + np.conjugate(Ez_plus) * Jzc))

        if m == 0:
            overlap_p_minus, overlap_c_minus = 0.0, 0.0
        else:
            overlap_p_minus = np.sum(weights * (np.conjugate(Ex_minus) * Jxp + np.conjugate(Ey_minus) * Jyp + np.conjugate(Ez_minus) * Jzp))
            overlap_c_minus = np.sum(weights * (np.conjugate(Ex_minus) * Jxc + np.conjugate(Ey_minus) * Jyc + np.conjugate(Ez_minus) * Jzc))

        eta_plus_sq = np.abs(overlap_p_plus)**2 / denom_plus + np.abs(overlap_p_minus)**2 / denom_minus
        eta_cross_sq = np.abs(overlap_c_plus)**2 / denom_plus + np.abs(overlap_c_minus)**2 / denom_minus
        eta_plus = np.sqrt(eta_plus_sq)
        eta_cross = np.sqrt(eta_cross_sq)
        etaps = np.append(etaps, eta_plus)
        etacs = np.append(etacs, eta_cross)

    return etaps, etacs, betas

def plot_form_factors(etaps, etacs, betas, mode_str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={"projection": "polar"})
    for ax in (ax1, ax2):
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlabel_position(90)
        ax.set_thetagrids(np.arange(0, 360, 90))

    ax1.plot(betas, etaps, lw=1.5)
    ax1.set_title(f'{mode_str}, ' + r'$\eta_+$')

    ax2.plot(betas, etacs, lw=1.5)
    ax2.set_title(f'{mode_str}, ' + r'$\eta_\times$')
    fig.tight_layout()
    fig.savefig(f'{mode_str}.png', dpi=600, bbox_inches="tight", pad_inches=0.03)


def main():
    # TE modes: p >= 1
    mode, m, n, p = 'TM', 0, 3, 0
    mode_str = mode + str(m) + str(n) + str(p)

    import time
    t1 = time.time()
    etaps, etacs, betas = form_factors(mode, m, n, p)
    t2 = time.time()
    print(f'Computed overlap factors for {mode}{m}{n}{p} in {t2 - t1} seconds.')

    plot_form_factors(etaps, etacs, betas, mode_str)
    t3 = time.time()
    print(f'Plotted overlap factors for {mode}{m}{n}{p} in {t3 - t2} seconds.')
    return

if __name__ == '__main__':
    main()