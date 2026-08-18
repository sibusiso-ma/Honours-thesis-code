# Hubble parameter vs z
# Sibusiso Mathebula
# 24/06/2026


import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# 2. Enlarge specific label types globally
plt.rcParams['axes.labelsize'] = 26   # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 24   # X-axis tick marks
plt.rcParams['ytick.labelsize'] = 24    # Y-axis tick marks
plt.rcParams['legend.fontsize'] = 24    # Legend text
plt.rcParams['axes.titlesize'] = 26



# Cosmological parameters
H0 = 67.36
Omega_m = 0.3153
Omega_r = 9.2e-5
Omega_k=0.0
Omega_l = 0.6847

def Hubble(z):
    return H0 * np.sqrt(
        Omega_r * (1 + z)**4 +
        Omega_m * (1 + z)**3 +
        Omega_k * (1 + z)**2 +
        Omega_l
    )

# Redshift range
z = np.linspace(0, 5, 1000)

# H(z)
Hz = Hubble(z)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z, Hz)
plt.xlabel("Redshift z")
plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
plt.title("Hubble Parameter vs Redshift")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(np.log10(z+1),np.log10( Hz))
plt.xlabel("Redshift z")
plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
plt.title("Hubble Parameter vs Redshift in log scale")
plt.show()



import numpy as np

# Cosmology parameters
Om = 0.3
Ol = 0.7
Ok = 0.0

c = 299792.458      # km/s
H0 = 70.0           # km/s/Mpc

def E(z):
   
    return Hubble(z) / H0

def f(z):
    return c / Hubble(z)

def comoving_trapezoidal(z, N=1000):
    z_vals = np.linspace(0, z, N+1)
    h = z / N

    integral = (h/2) * (
        f(z_vals[0]) +
        2*np.sum(f(z_vals[1:N])) +
        f(z_vals[N])
    )

    return  integral

print(comoving_trapezoidal(1.0))

def comoving_midpoint(z, N=1000):
    h = z / N
    z_mid = np.linspace(h/2, z - h/2, N)

    integral = h * np.sum(f(z_mid))

    return  integral

print(comoving_midpoint(1.0))

z_values = np.linspace(0, 5, 1000)

chi_trap = []
chi_mid = []

N = 200  # subdivisions for integration

for z in z_values:
    chi_trap.append(comoving_trapezoidal(z, N))
    chi_mid.append(comoving_midpoint(z, N))

chi_trap = np.array(chi_trap)
chi_mid = np.array(chi_mid)

plt.figure(figsize=(8,5))

plt.plot(z_values, chi_trap, label="Trapezoidal Rule")
plt.plot(z_values, chi_mid, label="Midpoint Rule", linestyle="--")

plt.xlabel("Redshift z")
plt.ylabel("Comoving Distance $\chi (z)$ [Mpc]")
plt.title("Comoving Distance vs Redshift")
plt.grid(True)
plt.legend()

plt.show()

error = np.abs(chi_trap - chi_mid)

plt.figure(figsize=(8,5))

plt.plot(z_values, error)

plt.xlabel("Redshift z")
plt.ylabel("|Difference| [Mpc]")
plt.title("Difference Between Trapezoidal and Midpoint Methods")
plt.grid(True)

plt.show()

# Low-z approximation
chi_lowz =  (c/H0)*z_values

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(z_values, chi_trap,
         label='Numerical (Trapezoidal)',
         linewidth=2)

plt.plot(z_values, chi_lowz,
         '--',
         label=r'Low-z Approximation: $\chi=\frac{cz}{H_0}$')

plt.xlabel("Redshift z")
plt.ylabel("Comoving Distance $\chi(z)$ [Mpc]")
plt.title("Comoving Distance vs Redshift")

plt.legend()
plt.grid(True)

plt.show()


def Omega_mass(z):
    return (Omega_m*((1+z)**3))/(E(z)**2)

def Growth_rate(z):
    return ((Omega_mass(z))**(0.55))


plt.figure(figsize=(8,5))
plt.plot(z_values, Growth_rate(z_values))
plt.xlabel("Redshift z")
plt.ylabel(r"$f(z)$ (dimensionelss)")
plt.title("Growth rate vs Redshift")
plt.show()

def trapezoidal_integral(func, a, b, N):
    z = np.linspace(a, b, N+1)
    y = func(z)
    h = (b-a)/N

    integral = h*(0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])
    return integral    # <-- Missing return statement


def integrand(z):
    return Growth_rate(z)/(1+z)

# Growth factor
def growth_factor(z, N=1000):
    I = trapezoidal_integral(integrand, 0, z, N)
    return np.exp(-I)

# Generate values
z_values = np.linspace(0, 5, 200)
D_values = np.array([growth_factor(z) for z in z_values])

# Plot
plt.figure(figsize=(8,5))
plt.plot(z_values, D_values)
plt.xlabel("Redshift z")
plt.ylabel("Growth factor D(z)")
plt.title("Growth Factor using Trapezoidal Rule")
plt.grid(True)
plt.show()



    
