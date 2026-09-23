# Redshift-space auto power spectrum multipoles for a single tracer
# Sibusiso Mathebula
#
# Implements the Computation section of the RSD project brief, steps 1-6:
#   1. Om(z), f(z) = Om(z)^gamma
#   2. Bias bA(z) = 1.34/D(z), number density nbar_A
#   3. Multipoles P0, P2, P4 at z = 0.5
#   4. Growth-index sensitivity: P2/P0 for gamma = 0.45, 0.55, 0.65
#   5. Shot-noise overlay on the monopole
#   6. Linear-scale cut k_max(z)

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

# Re-use the P(k,z), growth factor, and cosmological parameters already built
from Power_spectrum import power_spectrum, growth_factor, Om0, ns

# -----------------------------------------------------------------
# Step 1: Om(z) and the growth rate f(z) = Om(z)^gamma
# -----------------------------------------------------------------
def Om_of_z(z):
    """Matter density parameter at redshift z, for flat LCDM."""
    return Om0 * (1 + z) ** 3 / cosmo.efunc(z) ** 2

def growth_rate(z, gamma=0.55):
    """f(z) = Om(z)^gamma."""
    return Om_of_z(z) ** gamma


# -----------------------------------------------------------------
# Step 2: Bias model and number density for the DESI-like BGS tracer
# -----------------------------------------------------------------
def bias(z):
    """One-parameter bias model bA(z) = 1.34 / D(z) (DESI BGS-like)."""
    return 1.34 / growth_factor(z)

nbar_A = 1e-3   # (h^-1 Mpc)^-3, representative DESI BGS number density


# -----------------------------------------------------------------
# Step 6: linear-scale cut k_max(z) -- computed here since steps 3-5 use it
# -----------------------------------------------------------------
def kmax_of_z(z):
    """Conservative linear-regime cut, eq. (6) of the brief."""
    return 0.08 * (1 + z) ** (2 / (2 + ns))


# -----------------------------------------------------------------
# Multipoles P0, P2, P4 from bA, f and P(k,z)  [eqs. (8)-(10)]
# -----------------------------------------------------------------
def multipoles(k, z, gamma=0.55):
    """Return (P0, P2, P4) at redshift z for the given growth index gamma."""
    bA = bias(z)
    f = growth_rate(z, gamma)
    beta = f / bA
    Pk = power_spectrum(k, z)

    P0 = (1 + (2 / 3) * beta + (1 / 5) * beta ** 2) * bA ** 2 * Pk
    P2 = ((4 / 3) * beta + (4 / 7) * beta ** 2) * bA ** 2 * Pk
    P4 = (8 / 35) * beta ** 2 * bA ** 2 * Pk
    return P0, P2, P4


if __name__ == "__main__":

    z0 = 0.5
    kmax = kmax_of_z(z0)
    print(f"k_max(z={z0}) = {kmax:.4f} h/Mpc")
    print(f"bA(z={z0}) = {bias(z0):.3f}")
    print(f"f(z={z0}) = {growth_rate(z0):.3f}")

    # k-range required by the brief; restrict to the linear cut for the plots
    k_full = np.logspace(-3, np.log10(0.2), 300)
    k = k_full[k_full < kmax]

    # -----------------------------------------------------------------
    # Step 3: P0, P2, P4 at z = 0.5
    # -----------------------------------------------------------------
    P0, P2, P4 = multipoles(k, z0)

    plt.figure(figsize=(7, 5.5))
    plt.loglog(k, P0, label=r"$P_0(k)$ (monopole)")
    plt.loglog(k, P2, label=r"$P_2(k)$ (quadrupole)")
    plt.loglog(k, P4, label=r"$P_4(k)$ (hexadecapole)")

    # -----------------------------------------------------------------
    # Step 5: shot-noise level overlaid on the monopole
    # -----------------------------------------------------------------
    plt.axhline(1 / nbar_A, color="k", linestyle=":",
                label=r"shot noise $1/\bar n_A$")

    # k above which the monopole signal exceeds the shot noise
    above_noise = k[P0 > 1 / nbar_A]
    if len(above_noise) > 0:
        print(f"Signal exceeds shot noise for k > {above_noise.min():.4f} h/Mpc")

    plt.xlabel(r"$k \ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_\ell(k,z=0.5) \ [(\mathrm{Mpc}/h)^3]$")
    plt.title(f"RSD multipoles for a single tracer at $z={z0}$"
              f" ($k < k_{{\\max}}={kmax:.3f}\\,h\\,\\mathrm{{Mpc}}^{{-1}}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/RSD_multipoles_z0.5.png", dpi=200)
    print("Saved plot to /mnt/user-data/outputs/RSD_multipoles_z0.5.png")

    # -----------------------------------------------------------------
    # Step 4: growth-index sensitivity, P2/P0 for gamma = 0.45, 0.55, 0.65
    # -----------------------------------------------------------------
    plt.figure(figsize=(7, 5.5))
    for gamma in [0.45, 0.55, 0.65]:
        P0_g, P2_g, _ = multipoles(k, z0, gamma=gamma)
        plt.semilogx(k, P2_g / P0_g, label=fr"$\gamma = {gamma}$")

    plt.xlabel(r"$k \ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_2(k)/P_0(k)$")
    plt.title(f"Growth-index sensitivity of the quadrupole/monopole ratio"
              f" ($z={z0}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/RSD_growth_sensitivity.png", dpi=200)
    print("Saved plot to /mnt/user-data/outputs/RSD_growth_sensitivity.png")
