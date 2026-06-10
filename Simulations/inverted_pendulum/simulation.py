import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

G = 9.8
M = 1.0
m = 0.1
L = 0.5
CART_WIDTH = 0.4
CART_HEIGHT = 0.15


def control(state):
    x, x_dot, theta, theta_dot = state
    return 80 * theta + 12 * theta_dot + 20 * x + 5 * x_dot


def derivatives(t, state):
    x, x_dot, theta, theta_dot = state
    force = control(state)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    total_mass = M + m
    polemass_length = m * L
    temp = (force + polemass_length * theta_dot**2 * sin_t) / total_mass
    theta_acc = (G * sin_t - cos_t * temp) / (L * (4.0 / 3.0 - m * cos_t**2 / total_mass))
    x_acc = temp - polemass_length * theta_acc * cos_t / total_mass

    return [x_dot, x_acc, theta_dot, theta_acc]


state0 = [0.2, 0.0, 0.1, 0.0]
t = np.linspace(0, 20, 2500)
sol = solve_ivp(derivatives, [0, 20], state0, t_eval=t, max_step=0.02)

x = sol.y[0]
theta = sol.y[2]

pivot_y = CART_HEIGHT / 2
bob_x = x + L * np.sin(theta)
bob_y = pivot_y + L * np.cos(theta)

TRAIL_LENGTH = 150

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-0.1, 1.0)
ax.set_aspect('equal')
ax.set_title('Inverted Pendulum (Cart-Pole with Feedback Control)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axhline(0, color='black', lw=1)

cart = Rectangle((0, 0), CART_WIDTH, CART_HEIGHT, color='dimgray', ec='black')
ax.add_patch(cart)
rod, = ax.plot([], [], lw=2, color='steelblue')
bob, = ax.plot([], [], 'o', color='crimson', markersize=10)
trail, = ax.plot([], [], lw=1, color='crimson', alpha=0.6)


def update(frame):
    cart_x = x[frame] - CART_WIDTH / 2
    cart.set_xy((cart_x, 0))

    px = x[frame]
    py = pivot_y
    rod.set_data([px, bob_x[frame]], [py, bob_y[frame]])
    bob.set_data([bob_x[frame]], [bob_y[frame]])

    start = max(0, frame - TRAIL_LENGTH)
    trail.set_data(bob_x[start:frame], bob_y[start:frame])

    return cart, rod, bob, trail


ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)
plt.show()
