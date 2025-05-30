import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters 
g = 9.81                  # Gravitational acceleration due to gravity 
h = 1.0                   # Water depth 
a = 0.1                   # Wave amplitude 
lambda_ = 2 * np.pi       # Wavelength
k = 2 * np.pi / lambda_   # Wave number 
c = np.sqrt(g * h)        # Linear wave speed 
f = k * c                 # Wave frequency 

# Time domain
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

def compute_Ak(k_mode, c, h, g, lambda_, omega, s=0):
    '''Compute the second-order Fourier coefficients for a given mode.'''
    if abs(k_mode) != 2:
        return 0
    
    kh = k_mode * h
    sqrt_gh = np.sqrt(g * h)
    denom = np.sinh(k_mode**2 * h)

    # First term
    factor1 = (2 * np.pi**3 * h**2 * 1j * k_mode * (sqrt_gh * s - h * omega - c)) / (lambda_**2 * sqrt_gh * denom)
    inner1 = ((c - sqrt_gh * s + h * omega) * np.sinh(2 * h) -
              ((c - sqrt_gh * s + h * omega)**2) / (g * h * np.sinh(2 * h)**2))

    # Second term
    factor2 = (np.pi / (2 * 1j * sqrt_gh * denom))
    inner2 = (h * omega - 4 * np.pi * (c - sqrt_gh * s + h * omega) * (1 / np.tanh(2 * h)))

    # Final coefficient
    Ak = factor1 * inner1 + factor2 * inner2
    return Ak

def full_system_transformed(t, pos, omega, a, k, h, f, lambda_, g, A_k_dict):
    '''Defines the full ODE system in a moving reference fram including both first- and second order effects.'''
    X, Y = pos  

    sqrt_gh = np.sqrt(g * h)
    A0 = a * (f + k * h * omega) / np.sinh(k * h)

    # Clip large values of Y to avoid overflow
    Y_clip = np.clip(Y, -50, 50)

    # First-order velocity components
    dX = A0 * k * np.cos(X) * np.cosh(Y_clip) - omega * (Y_clip / k) - f
    dY = A0 * k * np.sin(X) * np.sinh(Y_clip)

    # Second-ordeer velocity components
    A1 = a**2 * sqrt_gh / (h * lambda_)
    for n, Ak in A_k_dict.items():
        factor = (n * k)**2 * (Y_clip / k) / h 
        factor = np.clip(factor, -50, 50)

        dX += A1 * k * (Ak * np.cosh(factor)).real
        dY += A1 * k * (Ak * np.sinh(factor)).real

    return [dX, dY]

def find_crossings(x, y, t, target):
    '''Find y-positions where the x-position of a particle trajectory corsses a specified vertical target.'''
    crossings = []
    for i in range(len(x) - 1):
        if (x[i] - target) * (x[i + 1] - target) < 0:
            # Interpolate in time and space to make sure this is the closest crossing
            t_cross = t[i] + (t[i + 1] - t[i]) * (target - x[i]) / (x[i + 1] - x[i])
            y_cross = y[i] + (y[i + 1] - y[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
            crossings.append(y_cross)
    return crossings

def solve_and_plot_transformed(start_values, omega, kryss_targets, *args):
    '''Solves the ODE system and plots resulting trajcetories  and  marks the Poincaré crossings'''
    plt.figure(figsize=(10, 6))
    # Iterate through the different initial positions
    for X0, Y0 in start_values:
        sol = solve_ivp(lambda t, y: full_system_transformed(t, y, omega, *args),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        plt.plot(sol.y[0], sol.y[1], label=f'Start: Y0={Y0:.2f}')                       
        # Mark crossings for each specified x-target
        for target in kryss_targets:
            y_cross = find_crossings(sol.y[0], sol.y[1], sol.t, target)
            if y_cross:
                plt.scatter([target]*len(y_cross), y_cross, marker='x', color='red')
    # Plot the phase portrait    
    plt.title(f'Particle trajectories in traveling frame for ω={omega}')
    plt.xlabel('X = kx - ft')
    plt.ylabel('Y = ky')
    plt.xlim(-np.pi/2, 1.5*np.pi)
    plt.ylim(0, h+0.2)
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'phase_portrait_omega_{omega}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Initial conditions
start_values = [(2, y0) for y0 in np.linspace(0.1, h, 12)]
cross_targets = [0, np.pi/2, np.pi]
vorticities = [-10.0, -5.0, -1.0, 1, 10.0]

# Iterate through each vorticity
for omega in vorticities:
    A_k_dict = {2: compute_Ak(2, c, h, g, lambda_, omega),
                -2: compute_Ak(-2, c, h, g, lambda_, omega)}
    solve_and_plot_transformed(start_values, omega, cross_targets, a, k, h, f, lambda_, g, A_k_dict)


