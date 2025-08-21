from scipy.special import jv, jvp
import numpy as np
from bessel import initialize_bessel_table, BesselJZeros, BesselJpZeros

def m_over_r_Jm(m, gamma, r):
    # safe m/r with no 1/r singularity
    return 0.5*gamma*(jv(m-1, gamma*r) + jv(m+1, gamma*r))

def k_z(p, L):
    return p*np.pi/L

def gamma_mn(family, m, n, a):
    if family.upper() == "TM":
        return BesselJZeros(m, n)/a
    elif family.upper() == "TE":
        return BesselJpZeros(m, n)/a
    else:
        raise ValueError("family must be 'TE' or 'TM'")

def omega_mnp(gamma_mn, kz_p):
    return np.sqrt(gamma_mn**2 + kz_p**2)

# def mnp_guard(m, n, p):
#     if not (isinstance(m, int) and isinstance(n, int) and isinstance(p, int)):
#         raise ValueError("m, n, p must be integers")
#     if m < 0 or n <= 0 or p < 0:
#         raise ValueError("m must be >= 0, n must be > 0, p must be >= 0")

# TM mode functions
def Er_TM_fn(m, n, p, a, L, sign="+"):
    g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
    def f(r, phi, z):
        ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
        return -(kz/g) * ang * np.sin(kz*z) * jvp(m, g*r)
    return f

def Ephi_TM_fn(m, n, p, a, L, sign="+"):
    g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
    def f(r, phi, z):
        ang = np.cos(m*phi) if sign == "+" else -np.sin(m*phi)
        return -(kz/g**2) * ang * np.sin(kz*z) * m_over_r_Jm(m, g, r)
    return f

def Ez_TM_fn(m, n, p, a, L, sign="+"):
    g  = gamma_mn("TM", m, n, a); kz = k_z(p, L)
    def f(r, phi, z):
        ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
        return ang * np.cos(kz*z) * jv(m, g*r)
    return f

# TE mode functions
def Er_TE_fn(m, n, p, a, L, sign="+", c=1.0):
    g  = gamma_mn("TE", m, n, a); kz = k_z(p, L); w = omega_mnp(g, kz)
    def f(r, phi, z):
        ang = np.cos(m*phi) if sign == "+" else -np.sin(m*phi)
        return 1j * w/g**2 * ang * np.sin(kz*z) * m_over_r_Jm(m, g, r)
    return f

def Ephi_TE_fn(m, n, p, a, L, sign="+", c=1.0):
    g  = gamma_mn("TE", m, n, a); kz = k_z(p, L); w = omega_mnp(g, kz)
    def f(r, phi, z):
        ang = np.sin(m*phi) if sign == "+" else np.cos(m*phi)
        return -1j * w/g * ang * np.sin(kz*z) * jvp(m, g*r)
    return f

def Ez_TE_fn(m, n, p, a, L, sign="+"):
    def f(r, phi, z):
        # TE has Ez = 0 in this convention
        return np.zeros_like(np.asarray(r, dtype=float))
    return f