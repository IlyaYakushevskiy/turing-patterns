import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import time
from scipy.ndimage.interpolation import rotate

np.random.seed(2)

# Diffusion
def laplacian(a, dx):
    return (-4 * a + np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
            + np.roll(a, +1, axis=1) + np.roll(a, -1, axis=1)) / (dx ** 2)

# Reaction functions
def Ra1(a, b): return a - a**3 - b + alpha
def Rb1(a, b): return (a - b) * beta

def Ra2(a, b): return - a*b*b + alpha*(1-a)
def Rb2(a, b): return + a*b*b - (alpha + beta)*b

def initialiser_normal(shape=(100, 100)):
    return (np.random.normal(loc=0, scale=0.05, size=shape),
            np.random.normal(loc=0, scale=0.05, size=shape))

def initialiser_2(shape):
    centre = int(shape[0] / 2)

    a = np.ones(shape)
    a[centre - 20:centre + 20, centre - 20:centre + 20] = 0.5
    a += np.random.normal(scale=0.05, size=shape)

    b = np.zeros(shape)
    b[centre-20:centre+20, centre-20:centre+20] = 0.25
    b += np.random.normal(scale=0.05, size=shape)

    return a, b

def average_rotate(a, degree):
    theta = 360 / degree
    a = np.mean([rotate(a, theta * i, reshape=False) for i in range(degree)], axis=0)
    return a

def initialiser_symmetry(shape):
    degree = 5
    a = np.random.normal(loc=0, scale=0.05, size=shape)
    b = np.random.normal(loc=0, scale=0.05, size=shape)

    return (
        average_rotate(a, degree), 
        average_rotate(b, degree)
    )

def one_time_step(a_current, b_current, Da, Db, Ra, Rb, dx=1, dt=0.1):
    La = laplacian(a_current, dx)
    Lb = laplacian(b_current, dx)
    delta_a = dt * (Da * La + Ra(a_current, b_current))
    delta_b = dt * (Db * Lb + Rb(a_current, b_current))
    a_current += delta_a
    b_current += delta_b
    return a_current, b_current

# Store snapshots every n frames
def run_animation(Da, Db, Ra, Rb, dt, N, initialiser, shape=(100, 100), dx=1, every=50):
    a, b = initialiser(shape)
    a_snapshots, b_snapshots = [], []
    for i in range(N):
        a, b = one_time_step(a, b, Da, Db, Ra, Rb, dx=dx, dt=dt)
        if i % every == 0:
            a_snapshots.append(a.copy())
            b_snapshots.append(b.copy())
    return a_snapshots, b_snapshots

# Animate a and b
def animate_concentrations(a_snapshots, b_snapshots, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(a_snapshots[0], cmap='jet', animated=True)
    im2 = ax2.imshow(b_snapshots[0], cmap='jet', animated=True)
    ax1.set_title('[a]')
    ax2.set_title('[b]')
    fig.suptitle(title)

    def update(frame):
        im1.set_array(a_snapshots[frame])
        im2.set_array(b_snapshots[frame])
        return im1, im2

    anim = animation.FuncAnimation(fig, update, frames=len(a_snapshots), interval=50, blit=True)
    anim.save(filename= f"results/{title}.gif", dpi=60, fps=10, writer='imagemagick')
    return anim

# SIMULATION 1: cool looking plots using t
# Da, Db, alpha, beta = 0.19, 0.05, 0.06, 0.062
dt, N = 1, 6000
t11 = time.time()
params = [
    [0.16, 0.08, 0.035, 0.065],
    [0.14, 0.06, 0.035, 0.065],
    [0.16, 0.08, 0.06, 0.062],
    [0.19, 0.05, 0.06, 0.062],
    [0.16, 0.08, 0.02, 0.055],
    [0.16, 0.08, 0.05, 0.065],
    [0.16, 0.08, 0.054, 0.063],
    [0.16, 0.08, 0.035, 0.06]
]
# for i, p in enumerate(params):
#     Da, Db, alpha, beta = p
#     a_snaps1, b_snaps1 = run_animation(Da, Db, Ra2, Rb2, dt, N, initialiser_2)
#     anim1 = animate_concentrations(a_snaps1, b_snaps1, f"Simulation 1.{i}: Ra2, Rb2")


figures = []
animations = []

#SQUARE INITIALISER
for i, p in enumerate(params):
    Da, Db, alpha, beta = p
    a_snaps1, b_snaps1 = run_animation(Da, Db, Ra2, Rb2, dt, N, initialiser_2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(a_snaps1[0], cmap='jet', animated=True)
    im2 = ax2.imshow(b_snaps1[0], cmap='jet', animated=True)
    ax1.set_title('[a]')
    ax2.set_title('[b]')
    title = f"Da: {Da}, Db: {Db}, alpha: {alpha}, beta: {beta} , square initialiser"
    fig.suptitle(title)

    def update(frame, i1=im1, i2=im2, a=a_snaps1, b=b_snaps1):
        i1.set_array(a[frame])
        i2.set_array(b[frame])
        return i1, i2

    anim = animation.FuncAnimation(fig, update, frames=len(a_snaps1), interval=50, blit=True)
    print(f"Animation {i} created.")
    figures.append(fig)
    anim.save(filename= f"results/{title}.gif", dpi=60, fps=10, writer='imagemagick')
    animations.append(anim)  # Keep reference alive

# NORMAL INITIALISER

params_normal = [
    [0.08,100, -0.05, 5],
    [0.05,100,-0.1, 5],
    [0.06,100,-0.06, 5]
]
for i, p in enumerate(params_normal):
    Da, Db, alpha, beta = p
    dt, N = 0.001, 20000
    a_snaps1, b_snaps1 = run_animation(Da, Db, Ra1, Rb1, dt, N, initialiser_normal)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(a_snaps1[0], cmap='jet', animated=True)
    im2 = ax2.imshow(b_snaps1[0], cmap='jet', animated=True)
    ax1.set_title('[a]')
    ax2.set_title('[b]')
    title = f"Da: {Da}, Db: {Db}, alpha: {alpha}, beta: {beta} , normal initialiser"
    fig.suptitle(title)

    def update(frame, i1=im1, i2=im2, a=a_snaps1, b=b_snaps1):
        i1.set_array(a[frame])
        i2.set_array(b[frame])
        return i1, i2

    dt, N = 0.001, 20000

    anim = animation.FuncAnimation(fig, update, frames=len(a_snaps1), interval=50, blit=True)
    print(f"Animation {i} created.")
    figures.append(fig)
    anim2 = animate_concentrations(a_snaps1, b_snaps1, title)
    animations.append(anim)  # Keep reference alive

# ---- ANIMATIONS WITH SYMMETRY ----

params_sym = [
    [0.16, 0.08, 0.035, 0.065],
    [0.16, 0.08, 0.06, 0.062],
    [0.19, 0.05, 0.06, 0.062],
    [1, 100, -0.01, 1], 
    [0.19, 0.05, 0.06, 0.062]
]

for i, p in enumerate(params_sym):
    Da, Db, alpha, beta = p
    a_snaps1, b_snaps1 = run_animation(Da, Db, Ra2, Rb2, dt, N, initialiser_symmetry)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(a_snaps1[0], cmap='jet', animated=True)
    im2 = ax2.imshow(b_snaps1[0], cmap='jet', animated=True)
    ax1.set_title('[a]')
    ax2.set_title('[b]')
    title = f"Da: {Da}, Db: {Db}, alpha: {alpha}, beta: {beta} , symmetrical initialiser"
    fig.suptitle(title)
    

    def update(frame, i1=im1, i2=im2, a=a_snaps1, b=b_snaps1):
        i1.set_array(a[frame])
        i2.set_array(b[frame])
        return i1, i2

    anim = animation.FuncAnimation(fig, update, frames=len(a_snaps1), interval=50, blit=True)
    anim.save(filename= f"results/{title}.gif", dpi=60, fps=10, writer='imagemagick')
    
    print(f"Animation {i} created.")
    figures.append(fig)
    animations.append(anim) 



t12 = time.time()
print(f'Time it took for the 1st animation to run: {t12 - t11}s')

# ---- SIM WITH D-GRADIENT ----
alpha, beta = -0.05, 5
Da_vals = np.linspace(0.01, 1.99, 100)
Da = np.ones((100, 1)) * Da_vals
Db = 100
dt, N = 0.001, 20000
t21 = time.time()
a_snaps2, b_snaps2 = run_animation(Da, Db, Ra1, Rb1, dt, N, initialiser_normal)
t22 = time.time()
print(f'Time it took for the 2nd animation to run: {t22 - t21}s')

# ---- ANIMATE ----
# anim1 = animate_concentrations(a_snaps1, b_snaps1, "Simulation 1: Ra2, Rb2")
anim2 = animate_concentrations(a_snaps2, b_snaps2, f"Da (0.01,1.99), Db {Db}, alpha {alpha}, beta {beta} , normal initialiser")
#plt.show()

