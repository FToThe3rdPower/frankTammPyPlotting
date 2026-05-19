"""
Frank-Tamm in energy form, in water. Shows that dN/(dx dE) is nearly flat
(~370 z^2 (1 - 1/(beta n)^2) photons / (eV cm)), with the only structure
coming from the wavelength dependence of n(E).
"""

import numpy as np
import matplotlib.pyplot as plt

ALPHA = 1.0 / 137.035999
HC_EV_NM = 1239.841984
HBARC_EV_CM = 1.97327e-5
COEFF = ALPHA / HBARC_EV_CM  # ~370 photons / (eV * cm) when (1 - 1/(beta n)^2) = 1


def n_water(lam_um):
    L = lam_um ** 2
    n2m1 = (
        5.672526103e-1 * L / (L - 5.085550461e-3)
        + 1.736581125e-1 * L / (L - 1.814938654e-2)
        + 2.121531502e-2 * L / (L - 2.617260739e-2)
        + 1.138493213e-1 * L / (L - 1.073888649e+1)
    )
    return np.sqrt(n2m1 + 1.0)


# Scenario
z = 1
beta = 1.0
medium_label = "20°C water"

# Energy grid covering 380-750 nm (so it matches the wavelength plot)
E_min = HC_EV_NM / 750.0   # ~ 1.653 eV
E_max = HC_EV_NM / 380.0   # ~ 3.542 eV
E_eV = np.linspace(E_min, E_max, 1000)

lam_nm = HC_EV_NM / E_eV
n = n_water(lam_nm * 1e-3)

# Frank-Tamm: photons / (eV cm)
dN_dxdE = COEFF * z**2 * (1.0 - 1.0 / (beta**2 * n**2))
dN_dxdE = np.where(beta * n > 1.0, dN_dxdE, np.nan)

# ----- Plot -----
fig, ax = plt.subplots(figsize=(6, 8))

y_top = 200.0

# Curve
ax.plot(E_eV, dN_dxdE, lw=2.6, color="#0b1d3a", zorder=3)
ax.fill_between(E_eV, 0, dN_dxdE, alpha=0.10, color="#0b1d3a", zorder=2)

ax.set_xlabel(r"Photon energy  $E$  [eV]", fontsize=12)
ax.set_ylabel(r"$\frac{d^2\mathsf{N}_\gamma}{dx\,dE}$   [$\frac{\mathsf{photons}}{\mathsf{cm} \cdot \mathsf{eV}}$]", fontsize=19)
ax.set_xlim(E_min, E_max)
ax.set_ylim(0, y_top)
ax.grid(alpha=0.3, zorder=1)



# Annotation pointing out flatness
ymin_curve = float(np.nanmin(dN_dxdE))
ymax_curve = float(np.nanmax(dN_dxdE))
diff = float(ymax_curve - ymin_curve)
pct = 100 * diff / ymin_curve
ax.text(
    0.97, 0.95,
    "Range across plot:\n"
    f"{ymin_curve:.1f}–{ymax_curve:.1f}={diff:.1f}\n"
    f"({pct:.1f}% variation)",
    transform=ax.transAxes, ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.92),
    zorder=5,
)

# Reference markers (same wavelengths as the wavelength plot, mapped to E)
# for lam_ref in [400.0, 550.0, 700.0]:
#     E_ref = HC_EV_NM / lam_ref
#     n_ref = float(n_water(lam_ref * 1e-3))
#     y_ref = COEFF * (1 - 1/n_ref**2)
#     ax.plot([E_ref], [y_ref], "o", color="#d2691e", ms=5, zorder=4)
#     ax.annotate(f"{y_ref:.1f}", xy=(E_ref, y_ref),
#                 xytext=(6, 6), textcoords="offset points",
#                 fontsize=9, color="#3d2208", zorder=4)
# Secondary x-axis: wavelength in nm
# def E_to_lam(E):
#    return np.where(np.abs(E) > 1e-9, HC_EV_NM / np.where(E == 0, 1e-9, E), np.nan)
# def lam_to_E(L):
#    return np.where(np.abs(L) > 1e-9, HC_EV_NM / np.where(L == 0, 1e-9, L), np.nan)
# secax = ax.secondary_xaxis("top", functions=(E_to_lam, lam_to_E))
# secax.set_xlabel(r"Wavelength  $\lambda$  (nm)", fontsize=11)

ax.set_title(
    f"Energy spectrum of Cherenkov radiation in {medium_label}\n"
    rf"$n \approx 1.33$, $z={z}$, $\beta={beta:g}$",
    fontsize=12,
)

plt.tight_layout()
print(f"\nMin: {ymin_curve:.2f}, Max: {ymax_curve:.2f}, variation: {pct:.2f}%")
