import numpy as np
from scipy.optimize import root

# Parametere
g = 9.81
alpha = 0.3
h = 1.0
sigma = 1.0

B_10 = 0.01
B_01 = 0.01

def k_mag(k, l):
    '''Compute the magnitude of the wavenumber vector'''
    return np.sqrt(k**2 + l**2)

def compute_gamma(k, l):
    '''Compute the gamma value based on the wavenumbers k and l'''
    return np.sqrt(k_mag(k, l)**2 - alpha**2)

def phi_kl(k, l, z):
    '''Vertical shape function for the velocity field'''
    gamma = compute_gamma(k, l)
    return z / h if gamma == 0 else np.sinh(gamma * z) / np.sinh(gamma * h)

def dphi_dz(z, k, l):
    '''Derivative of the vertical shape function with respect to z'''
    gamma = compute_gamma(k, l)
    return 1 / h if gamma == 0 else gamma * np.cosh(gamma * z) / np.sinh(gamma * h)

def eta(x, y):
    '''Compute the surface elevation eta at a given position using known Fourier coefficients B_10 and B_01'''
    return np.real(B_10 * np.exp(1j * x) + B_01 * np.exp(1j * y))

def lap_eta(x, y):
    '''Compute the Laplacian of the surface elevation eta at a given position'''
    return np.real(-B_10 * np.exp(1j * x) - B_01 * np.exp(1j * y))

def dx_eta(x):
    '''Compute the partial derivative of the surface elevation eta with respect to x at a given position'''
    return np.real(1j * B_10 * np.exp(1j * x))

def dy_eta(y):
    '''Compute the partial derivative of the surface elevation eta with respect to y at a given position'''
    return np.real(1j * B_01 * np.exp(1j * y))

def U0(z, c1, c2):
    '''Compute the initial velocity field at a given height z using Fourier coefficients c1 and c2'''
    return np.array([c1 * np.cos(alpha * (z - h)) + c2 * np.sin(alpha * (z - h)), 
                     -c1 * np.sin(alpha * (z - h)) + c2 * np.cos(alpha * (z - h)),
                     0])

def u(x, y, z, c1, c2):
    '''Compute the velocity field at a given position (x, y) and height z using Fourier coefficients c1 and c2'''
    du_eta = np.array([dx_eta(x), dy_eta(y)])
    u0 = U0(h, c1, c2)
    A_10 = 1j * np.dot(u0[:2], du_eta)

    # Fourier coefficients for the vertical shape functions
    phi10 = phi_kl(1, 0, z)
    phi01 = phi_kl(0, 1, z)
    dphi10 = dphi_dz(z, 1, 0)
    dphi01 = dphi_dz(z, 0, 1)

    # Velocity field components
    u1 = np.real(1j * A_10 * phi10 * np.exp(1j * x))
    u2 = np.real(1j * A_10 * phi01 * np.exp(1j * y))
    u3 = np.real(A_10 * (dphi10 * np.exp(1j * x) + dphi01 * np.exp(1j * y)))

    return np.array([u1, u2, u3])

def dynamic_condition(x, y, z, c1, c2):
    '''Dynamic boundary condition for the surface elevation'''
    u0 = U0(z, c1, c2)
    uf = u(x, y, z, c1, c2)
    return np.dot(u0, uf) + g * eta(x, y) - sigma * lap_eta(x, y)

def system_to_solve(c):
    '''System of equations to solve for the coefficients c1 and c2'''
    c1, c2 = c
    eq1 = dynamic_condition(np.pi/2, 0, h, c1, c2)  # matches B_10
    eq2 = dynamic_condition(0, np.pi/2, h, c1, c2)  # matches B_01
    return [eq1, eq2]

# Solve the system of equations
sol = root(system_to_solve, [1.0, 1.0])
print("Solution c1, c2:", sol.x)
print("Residual:", system_to_solve(sol.x))
