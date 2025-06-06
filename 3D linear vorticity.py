import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import least_squares

# Parameters 
g = 9.81            # Gravitational acceleration
alpha = 0.3         # Beltrami parameter
h = 1.0             # Water depth 
sigma = 1           # Surface tension 

B_10 = 0.01    # k=1, l=0
B_01 = 0.01    # k=0, l=1

def k_mag (k,l):
    '''Compute the magnitude of the wavenumber vector'''
    return np.sqrt(k**2 + l**2)

def compute_gamma (k,l):
    '''Compute the gamma value based on the wavenumbers k and l'''
    return np.sqrt(k_mag(k,l)**2 - alpha**2)

def phi_kl(k,l,z):
    '''Vertical shape function for the velocity field'''
    gamma = compute_gamma(k, l)
    if gamma == 0:
        return z / h
    else:
        return np.sinh(gamma * z) / np.sinh(gamma * h)

def dphi_dz(z, k, l, h):
    '''Derivative of the vertical shape function with respect to z'''
    gamma = compute_gamma(k, l)
    if gamma == 0:
        return 1 / h 
    else:
        return gamma * np.cosh(gamma * z) / np.sinh(gamma * h)

def eta (pos):
    '''Compute the surface elevation eta at a given position using known Fourier coefficients B_10 and B_01'''
    x,y,_ = pos
    return np.real(B_10 * np.exp(1j* x) + B_01 * np.exp(1j * y))

def dx_eta (pos):
    '''Compute the partial derivative of the surface elevation eta with respect to x at a given position'''
    x, _, _ = pos
    return np.real(1j * B_10 * np.exp(1j * x))

def dy_eta (pos):
    '''Compute the partial derivative of the surface elevation eta with respect to y at a given position'''
    _, y, _ = pos
    return np.real(1j * B_01 * np.exp(1j * y))

def delta_eta (pos):
    '''Compute the gradient of the surface elevation eta at a given position'''
    x, y, _ = pos
    return np.real(-B_10*np.exp(1j*x)-B_01*np.exp(1j*y))

def U0 (z, c1, c2):
    '''Compute the initial velocity field at a given height z using Fourier coefficients c1 and c2'''
    return [c1*np.cos(alpha*(z-h))+c2*np.sin(alpha*(z-h)), -c1*np.sin(alpha*(z-h))+c2*np.cos(alpha*(z-h)), 0]

"""
def kinematic_condition(pos, c1, c2):
    '''Kinematic boundary condition for the surface elevation'''
    u_vec = U0(h, c1, c2)
    u3_xyh = u_vec[0] * dx_eta(pos) + u_vec[1] * dy_eta(pos)
    return u3_xyh
"""

def kinematic_condition(pos, c1, c2):
    '''Return u3 at surface z = h'''
    return u(pos, c1, c2)[2]  # u3 at surface

def u_hat (c1, c2):
    '''Compute the velocity field in Fourier space'''
    A_10 = 1j * kinematic_condition([np.pi/2,0,h], c1, c2)
    A_01 = 1j * kinematic_condition([0,np.pi/2,h], c1, c2)
    return (A_10, A_01)


def u(pos, c1, c2):
    x, y, z = pos
    A_10, A_01 = u_hat(c1, c2)

    phi10 = phi_kl(1, 0, z)
    phi01 = phi_kl(0, 1, z)
    dphi10 = dphi_dz(z, 1, 0, h)
    dphi01 = dphi_dz(z, 0, 1, h)

    # k=1, l=0 => u1: i*k*phi, u2: 0, u3: dphi
    # k=0, l=1 => u1: 0, u2: i*l*phi, u3: dphi

    u1 = np.real(1j * 1 * A_10 * phi10 * np.exp(1j * x))         # x-retning
    u2 = np.real(1j * 1 * A_01 * phi01 * np.exp(1j * y))         # y-retning
    u3 = np.real(A_10 * dphi10 * np.exp(1j * x) + A_01 * dphi01 * np.exp(1j * y))  # z-retning

    return (u1, u2, u3)

def dynamic_condition(pos, c1, c2):
    x, y, z = pos
    u0 = U0(z, c1, c2)
    uval = u([x, y, z], c1, c2)
    return u0[0]*uval[0] + u0[1]*uval[1] + g * eta([x, y, z]) - sigma * delta_eta([x, y, z])


def hat_dynamic_condition(c):
    '''Compute the dynamic boundary condition in Fourier space'''
    c1, c2 = c
    D_10 = dynamic_condition([0,np.pi/2,h], c1, c2)
    D_01 = dynamic_condition([np.pi/2,0,h], c1, c2)
    return [D_10, D_01]


start_positions = ((1, 1, 1))
c = least_squares(hat_dynamic_condition, [1.0, 1.0])
print(f'Da får vi at c blir:', c)

"""
# ------- NYTT --------
def kappa(k_mag, alpha, h):
    if alpha > k_mag:
        val = np.sqrt(alpha**2 - k_mag**2)
        return val * 1 / np.tan(val * h)
    elif np.isclose(alpha, k_mag):
        return 1 / h
    else:
        val = np.sqrt(k_mag**2 - alpha**2)
        return val / np.tanh(val * h)

def rho_dispersion(c1, c2, k_vec):
    c = np.array([c1, c2])
    k = np.array(k_vec)
    k_perp = np.array([-k[1], k[0]])
    k_mag_sq = np.dot(k, k)
    c_dot_k = np.dot(c, k)
    c_dot_k_perp = np.dot(c, k_perp)
    kappa_val = kappa(np.sqrt(k_mag_sq), alpha, h)

    rho = g + sigma * k_mag_sq \
        - (c_dot_k ** 2) / k_mag_sq * kappa_val \
        + alpha * (c_dot_k * c_dot_k_perp) / k_mag_sq
    return rho

def hat_dispersion_condition(c):
    return [
        rho_dispersion(c[0], c[1], [1, 0]),
        rho_dispersion(c[0], c[1], [0, 1])
    ]

# 💡 Ny funksjon som kombinerer begge:
def combined_condition(c):
    return np.concatenate((hat_dynamic_condition(c), hat_dispersion_condition(c)))

# 🧠 Løs med least_squares
res = least_squares(combined_condition, [1, 1])
c_opt = res.x
print("Optimalisert c:", c_opt)

# 🔍 Evaluer restledd for innsikt
print("Dynamic residual:", hat_dynamic_condition(c_opt))
print("Dispersion residual:", hat_dispersion_condition(c_opt))

# 📊 Eksempel: beregn rho for å bruke videre
rho1 = rho_dispersion(c_opt[0], c_opt[1], [1, 0])
rho2 = rho_dispersion(c_opt[0], c_opt[1], [0, 1])
print("rho1 =", rho1)
print("rho2 =", rho2)







def kappa(k_mag, alpha, h):
    if alpha > k_mag:
        val = np.sqrt(alpha**2 - k_mag**2)
        return val * 1 / np.tan(val * h) 
    elif np.isclose(alpha, k_mag):
        return 1 / h
    else:
        val = np.sqrt(k_mag**2 - alpha**2)
        return val / np.tanh(val * h)


def rho_dispersion(c1, c2, k_vec):
    c = np.array([c1, c2])
    k = np.array(k_vec)
    k_perp = np.array([-k[1], k[0]])
    k_mag_sq = np.dot(k, k)
    c_dot_k = np.dot(c, k)
    c_dot_k_perp = np.dot(c, k_perp)
    kappa_val = kappa(np.sqrt(k_mag_sq), alpha, h)

    rho = g + sigma * k_mag_sq \
          - (c_dot_k ** 2) / k_mag_sq * kappa_val \
          + alpha * (c_dot_k * c_dot_k_perp) / k_mag_sq
    return rho

k1 = [1, 0]
k2 = [0, 1]
rho1 = rho_dispersion(c[0], c[1], k1)
rho2 = rho_dispersion(c[0], c[1], k2)
print("rho1 = ", rho1)
print("rho2 = ", rho2)

def hat_dispersion_condition(c):
    return [
        rho_dispersion(c[0], c[1], [1, 0]),
        rho_dispersion(c[0], c[1], [0, 1])
    ]

c = fsolve(hat_dispersion_condition, [1, 1])
print("c =", c)
print("rho1 =", rho_dispersion(c[0], c[1], [1, 0]))
print("rho2 =", rho_dispersion(c[0], c[1], [0, 1]))
"""