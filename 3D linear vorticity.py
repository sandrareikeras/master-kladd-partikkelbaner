import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Parameters 
k = 1.0             # Wavenumber in x
l = 0.5             # Wavenumber in y
alpha = 0.3         # Beltrami parameter
h = 1.0             # Water depth
A = 1.0             # Wave amplitude
omega = 1.0         # Wave frequency

# Time domain
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

k_mag = np.sqrt(k**2 + l**2)
gamma = np.sqrt(k_mag**2 - alpha**2)

def phi(z):
    '''Vertical shape function for the velocity field'''
    return np.sinh(gamma * z) / np.sinh(gamma * h)

def dphi(z):
    '''Derivative of the vertical shape function for the velocity field'''
    return gamma * np.cosh(gamma * z) / np.sinh(gamma * h)

def velocity_field(t, pos):
    '''Compute the velocity components, the flow is based on the Fourier ansatz for a Beltrami flow'''
    x, y, z = pos
    phase = np.exp(1j * (k * x + l * y - omega * t))
    u1_hat = 1j * (k / k_mag) * A * ( (k / k_mag) * dphi(z) + alpha * phi(z) )
    u2_hat = 1j * (l / k_mag) * A * ( (l / k_mag) * dphi(z) - alpha * phi(z) )
    u3_hat = A * phi(z)
    u1 = np.real(u1_hat * phase)
    u2 = np.real(u2_hat * phase)
    u3 = np.real(u3_hat * phase)
    return [u1, u2, u3]

def find_crossings_3d(x, y, z, t, target):
    '''Find (y,z) coordinates where the x-position of a particle trajectory crosses a specified x target value'''
    crossings = []
    for i in range(len(x) - 1):
        if (x[i] - target) * (x[i+1] - target) < 0:
            # Interpolate in time and space to make sure this is the closest crossing
            t_cross = t[i] + (t[i+1] - t[i]) * (target - x[i]) / (x[i+1] - x[i])
            y_cross = y[i] + (y[i+1] - y[i]) * (t_cross - t[i]) / (t[i+1] - t[i])
            z_cross = z[i] + (z[i+1] - z[i]) * (t_cross - t[i]) / (t[i+1] - t[i])
            crossings.append((y_cross, z_cross))
    return np.array(crossings)

def solve_and_plot_3d(start_positions, kryss_targets, t_span, t_eval):
    '''Solves The ODE system and plots resulti9ng trajectories and  marks the Poincaré crossings'''
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    poincare_colors = {0: 'red', np.pi/2: 'blue', np.pi: 'green'}       # Define Poincaré colors
    proxy_handles = []                                                  # Proxy handles for legend
    planes_added = set()                                                # Keep track of which plane has been added to the legend

    # Iterate through the different initial positions
    for X0, Y0, Z0 in start_positions:
        sol = solve_ivp(velocity_field, t_span, [X0, Y0, Z0], t_eval=t_eval, method='RK45')
        ax.plot(sol.y[0], sol.y[1], sol.y[2], alpha=0.6, label=f'Start: X0={X0:.2f}, Y0={Y0:.2f}, Z=={Z0:.2f}')

        # Mark crossings for each specified x-target
        for target in kryss_targets:
            crossings = find_crossings_3d(sol.y[0], sol.y[1], sol.y[2], sol.t, target)
            if len(crossings) > 0:
                color = poincare_colors.get(target, 'black')
                ax.scatter([target]*len(crossings), crossings[:,0], crossings[:,1],
                           marker='x', color=color, s=50)
                if target not in planes_added:
                    proxy = mlines.Line2D([], [], color=color, marker='x', linestyle='None',
                                          markersize=8, label=f'Poincaré at x={target:.2f}')
                    proxy_handles.append(proxy)
                    planes_added.add(target)

    # Add legend for Poincaré planes
    handles, labels = ax.get_legend_handles_labels()
    handles += proxy_handles
    ax.legend(handles=handles, loc='upper left', fontsize=9)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Particle Trajectories with Poincaré Crossings')

    plt.tight_layout()
    plt.show()

# Initial conditions
X_range = np.linspace(0, 0.5, 2)
Y_range = np.linspace(0, 0.5, 2)  
Z_range = np.linspace(0.1, 0.5, 2)  
start_positions = [[X0, Y0, Z0] for X0 in X_range for Y0 in Y_range for Z0 in Z_range]
kryss_targets = [0, np.pi/2, np.pi]

# Plot for each starting value
solve_and_plot_3d(start_positions, kryss_targets, t_span, t_eval)
