import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



# 2. Enlarge specific label types globally
plt.rcParams['axes.labelsize'] = 18       # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 14      # X-axis tick marks
plt.rcParams['ytick.labelsize'] = 14      # Y-axis tick marks
plt.rcParams['legend.fontsize'] = 14      # Legend text
plt.rcParams['axes.titlesize'] = 16

# -----------------------------
# Cosmological parameters
# -----------------------------
Omega_m0 = 0.3
Omega_L0 = 0.7

# Scale factor range
a_i = 0.01
a_f = 1.0

# Convert to ln(a)
x_i = np.log(a_i)
x_f = np.log(a_f)

# -----------------------------
# Define coupled ODE system
# y[0] = f
# y[1] = Omega_m
# -----------------------------
def growth_system(x, y):

    f, Omega_m = y

    df_dx = (3/2)*Omega_m - f*(0.5*(4 - 3*Omega_m + f))

    dOmega_dx = -3*(1 - Omega_m)*Omega_m

    return [df_dx, dOmega_dx]


# -----------------------------
# Initial conditions
# Matter domination:
# f(ai)=1
# Omega_m(ai)≈1
# -----------------------------
y0 = [1.0, 0.999]


# Solve ODE
solution = solve_ivp(
    growth_system,
    [x_i, x_f],
    y0,
    dense_output=True,
    rtol=1e-8,
    atol=1e-10
)


# Generate scale factor values
a_values = np.linspace(a_i, a_f, 500)
x_values = np.log(a_values)

# Numerical solution
f_numeric, Omega_numeric = solution.sol(x_values)


# -----------------------------
# Approximate growth rate
# f = Omega_m^0.55
# -----------------------------

# Present-day Omega_m evolution
Omega_m_curve = (
    Omega_m0*a_values**(-3)
    /
    (Omega_m0*a_values**(-3) + Omega_L0)
)

f_approx = Omega_m_curve**0.55


# -----------------------------
# Plot comparison
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    a_values,
    f_numeric,
    label="Numerical solution of coupled ODEs",
    linewidth=2
)

plt.plot(
    a_values,
    f_approx,
    "--",
    label=r"$f(a)=\Omega_m(a)^{0.55}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel("Growth rate $f(a)$")

plt.title("Matter Growth Rate Evolution")

plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# Plot Omega_m(a)
# -----------------------------

plt.figure(figsize=(8,5))

# Numerical solution
plt.plot(
    a_values,
    Omega_numeric,
    label=r"Numerical $\Omega_m(a)$",
    linewidth=2
)

# Analytical Lambda-CDM solution
plt.plot(
    a_values,
    Omega_m_curve,
    "--",
    label=r"$\Omega_m(a)=\frac{\Omega_{m0}a^{-3}}{\Omega_{m0}a^{-3}+\Omega_{\Lambda0}}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel(r"$\Omega_m(a)$")

plt.title("Matter Density Parameter Evolution")

plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# Fractional difference
# -----------------------------

fd = (f_approx - f_numeric) / f_numeric


# Convert to percentage
fd_percent = 100 * fd


# -----------------------------
# Plot fractional difference
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    a_values,
    fd_percent,
    linewidth=2
)

plt.axhline(
    1,
    linestyle="--",
    label="1%"
)

plt.axhline(
    -1,
    linestyle="--",
    label="-1%"
)

plt.xlabel("Scale factor $a$")
plt.ylabel(
    r"$f_d(\%)=\frac{f_{\rm approx}-f_{\rm numerical}}{f_{\rm numerical}}\times100$"
)

plt.title("Fractional Difference Between Growth Rate Solutions")

plt.legend()
plt.grid(True)

plt.show()


def growth_system(x, y):
    
    f, Omega_m = y

    # Correct growth equation
    df_dx = (
        1.5*Omega_m
        - f*(0.5*(4 - 3*Omega_m) + f)
    )

    # Matter density evolution
    dOmega_dx = -3*(1 - Omega_m)*Omega_m

    return [df_dx, dOmega_dx]

a_i = 0.01

Omega_m_i = (
    Omega_m0*a_i**(-3)
    /
    (Omega_m0*a_i**(-3)+Omega_L0)
)

y0 = [1.0, Omega_m_i]

Omega_m_i = 0.999

# Solve ODE
solution = solve_ivp(
    growth_system,
    [x_i, x_f],
    y0,
    dense_output=True,
    rtol=1e-8,
    atol=1e-10
)


# Generate scale factor values
a_values = np.linspace(a_i, a_f, 500)
x_values = np.log(a_values)

# Numerical solution
f_numeric, Omega_numeric = solution.sol(x_values)


# -----------------------------
# Approximate growth rate
# f = Omega_m^0.55
# -----------------------------

# Present-day Omega_m evolution
Omega_m_curve = (
    Omega_m0*a_values**(-3)
    /
    (Omega_m0*a_values**(-3) + Omega_L0)
)

f_approx = Omega_m_curve**0.55


# -----------------------------
# Plot comparison
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    a_values,
    f_numeric,
    label="Numerical solution of f",
    linewidth=2
)

plt.plot(
    a_values,
    f_approx,
    "--",
    label=r"$f(a)=\Omega_m(a)^{0.55}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel("Growth rate $f(a)$")

plt.title("Matter Growth Rate Evolution")

plt.legend()
plt.grid(True)

plt.show()


plt.plot(
    a_values,
    Omega_m_curve,
    "--",
    label=r"$\Omega_m(a)=\frac{\Omega_{m0}a^{-3}}{\Omega_{m0}a^{-3}+\Omega_{\Lambda0}}$",
    linewidth=2
)

plt.xlabel("Scale factor $a$")
plt.ylabel(r"$\Omega_m(a)$")

plt.title("Matter Density Parameter Evolution")

plt.legend()
plt.grid(True)

plt.show()

d = (f_approx - f_numeric) / f_numeric


# Convert to percentage
fd_percent = 100 * fd


# -----------------------------
# Plot fractional difference
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    a_values,
    fd_percent,
    linewidth=2
)

plt.axhline(
    1,
    linestyle="--",
    label="1%"
)

plt.axhline(
    -1,
    linestyle="--",
    label="-1%"
)

plt.xlabel("Scale factor $a$")
plt.ylabel(
    r"$f_d(\%)=\frac{f_{\rm approx}-f_{\rm numerical}}{f_{\rm numerical}}\times100$"
)

plt.title("Fractional Difference Between Growth Rate Solutions")

plt.legend()
plt.grid(True)

plt.show()
