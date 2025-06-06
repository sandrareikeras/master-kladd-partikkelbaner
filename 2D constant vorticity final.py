import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Parameters 
g = 9.81                  # Gravitational acceleration
h = 1.0                   # Water depth 
a = 0.1                   # Wave amplitude 
lambda_ = 2 * np.pi       # Wavelength
k = 2 * np.pi / lambda_   # Wave number 
c = np.sqrt(g * h)        # Linear wave speed 
f = k * c                 # Wave frequency 
s = 0.5                   # Integration constant

# Time domain
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

def compute_Ak(k_mode, c, h, g, lambda_, omega, s):
    '''Compute the second-order Fourier coefficients for a given mode.'''
    if abs(k_mode) != 2:
        return 0
    
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

def full_system_transformed(pos, omega, a, k, h, f, lambda_, g, A_k_dict):
    '''Defines the full ODE system in a moving reference fram including both first- and second order effects.'''
    X, Y = pos  

    sqrt_gh = np.sqrt(g * h)
    A0 = a * (f + k * h * omega) / np.sinh(k * h)

    # Clip large values of Y to avoid overflow
    Y_clip = np.clip(Y, -100, 100)

    # First-order velocity components
    dX = A0 * k * np.cos(X) * np.cosh(Y_clip) - omega * (Y_clip / k) - f
    dY = A0 * k * np.sin(X) * np.sinh(Y_clip)

    # Second-ordeer velocity components
    A1 = a**2 * sqrt_gh / (h * lambda_)
    for n, Ak in A_k_dict.items():
        factor = (n * k)**2 * (Y_clip / k) / h 
        factor = np.clip(factor, -100, 100)

        dX += A1 * k * (Ak * np.cosh(factor)).real
        dY += A1 * k * (Ak * np.sinh(factor)).real

    return [dX, dY]

def solve_and_plot_trajcetories(start_values, omega, kryss_targets, *args):
    '''Solves the ODE system, plots resulting trajcetories and marks the Poincaré crossings'''
    plt.figure(figsize=(11, 7))

    for X0, Y0 in start_values:
        line_color = color_map_combo[(X0, Y0)]
        sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        # Plot trajectory
        plt.plot(sol.y[0], sol.y[1], label=f'Start: X0={X0:.2f}, Y0={Y0:.2f}', color=line_color)

    # Plot settings
    plt.title(f'Particle trajectories in a travelling frame for ω={omega}')
    plt.xlabel(r'$X = kx - ft$')
    plt.ylabel(r'$Y = ky$')
    plt.xlim(-lambda_*0.75, lambda_*1.75)
    plt.ylim(0, h+0.25)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'phase_portrait_omega_{omega}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

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

def collect_crossings(start_values, omega, kryss_targets, A_k_dict):
    '''Collects Poincaré crossings for each target value in kryss_targets.'''
    summary = {target: {'Y0': [], 'Y_cross': []} for target in cross_targets}
    for X0, Y0 in start_values:
        sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        for target in kryss_targets:
            y_cross = find_crossings(sol.y[0], sol.y[1], sol.t, target)
            if y_cross:
                summary[target]['Y0'].extend([Y0] * len(y_cross))
                summary[target]['Y_cross'].extend(y_cross)
    return summary

def plot_crossings_per_omega_subplots(all_crossing_summaries, cross_targets, poincare_colors):
    '''Plots a 3-panel subplot figure per vorticity, one panel per Poincaré target.'''
    for omega, summary in all_crossing_summaries.items():
        fig, axes = plt.subplots(1, len(cross_targets), figsize=(18, 5), sharey=True)

        for i, target in enumerate(cross_targets):
            ax = axes[i]
            Y0s = summary[target]['Y0']
            Ycross = summary[target]['Y_cross']
            if Y0s:
                ax.scatter(Y0s, Ycross,
                           alpha=0.7,
                           s=12,
                           color=poincare_colors.get(target, 'black'),
                           label=f'X = {target:.2f}')
            ax.set_title(f'X = {target:.2f}')
            ax.set_xlabel('Initial $Y_0$')
            if i == 0:
                ax.set_ylabel('Poincaré Crossing $Y$')
            ax.grid()
            ax.legend()

        plt.suptitle(f'Poincaré Crossings by Section for ω = {omega:.2f}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f'poincare_crossings_omega_{omega:.2f}_subplots.pdf', dpi=300)
        plt.show()
        plt.close(fig)


# Initial conditions
start_values = [(x0, y0) for x0 in np.linspace(0, lambda_, 5) for y0 in np.linspace(0.1, h, 5)]
start_values_crossings = [(x0, y0) for x0 in np.linspace(0, lambda_, 10) for y0 in np.linspace(0.1, h, 10)]

cross_targets = [0, lambda_/2, lambda_]
poincare_colors = {cross_targets[0]: 'red', cross_targets[1]: 'blue', cross_targets[2]: 'green'}
vorticities = [-10.0, -4.0, 1]
all_crossing_summaries = {}

cmap = plt.get_cmap('Dark2')  
unique_combinations = list(sorted(set(start_values)))  # All unique (X0, Y0) pairs
color_map_combo = {combo: cmap(i % cmap.N) for i, combo in enumerate(unique_combinations)}

# Main loop to solve the system for each vorticity and plot results
for omega in vorticities:
    A_k_dict = {2: compute_Ak(2, c, h, g, lambda_, omega, s),
                -2: compute_Ak(-2, c, h, g, lambda_, omega, s)}
    solve_and_plot_trajcetories(start_values, omega, cross_targets, A_k_dict, a, k, h, f, lambda_, g)
    summary = collect_crossings(start_values_crossings, omega, cross_targets, A_k_dict)
    all_crossing_summaries[omega] = summary

plot_crossings_per_omega_subplots(all_crossing_summaries, cross_targets, poincare_colors)


# Define a dictionary with vorticity: start_values mapping
start_values_dict = {}
more_vorticities = np.linspace(-30, 1, 10)  # More vorticities for a smoother transition
all_trajectories = []

for omega in more_vorticities:
    if omega < -30:
        y_vals = np.linspace(0, 0.3, 2)
    elif omega < -20:
        y_vals = np.linspace(0.2, 0.5, 2)
    elif omega <-10:
        y_vals = np.linspace(0.3, 0.7, 2)
    elif omega < -5:
        y_vals = np.linspace(0.4, 0.9, 2)
    elif omega < -1:
        y_vals = np.linspace(0.5, h, 2)
    else:
        y_vals = np.linspace(0.6, h, 2)

    x_vals = np.linspace(-1, 1, 3)
    start_values = [(x0, y0) for x0 in x_vals for y0 in y_vals]
    start_values_dict[omega] = start_values

for omega in more_vorticities:
    A_k_dict = {2: compute_Ak(2, c, h, g, lambda_, omega, s),
                -2: compute_Ak(-2, c, h, g, lambda_, omega, s)}
    current_start_values = start_values_dict[omega]
    
    for X0, Y0 in current_start_values:
        sol = solve_ivp(lambda t, y: full_system_transformed(y, omega, a, k, h, f, lambda_, g, A_k_dict),
                        t_span, [X0, Y0], t_eval=t_eval, method='RK45')
        all_trajectories.append({
            'omega': omega,
            'X': sol.y[0],
            'Y': sol.y[1],
            'X0': X0,
            'Y0': Y0})
plt.figure(figsize=(12, 8))

for traj in all_trajectories:
    plt.plot(traj['X'], traj['Y'], linewidth=0.7, label=f'ω={traj["omega"]:.2f}, X0={traj["X0"]:.2f}, Y0={traj["Y0"]:.2f}')

plt.title('Particle trajectories for multiple vorticity values')
plt.xlabel(r'$X = kx - ft$')
plt.ylabel(r'$Y = ky$')
plt.xlim(-np.pi, np.pi)
plt.ylim(0, h + 0.2)
plt.grid()
plt.legend(fontsize=4)
plt.tight_layout()
plt.savefig('all_vorticities_custom_startvalues.pdf', dpi=300, bbox_inches='tight')
plt.show()

