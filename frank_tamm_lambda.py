"""
Frank-Tamm spectrum, wavelength form, with physically-motivated
visible-spectrum background (Bruton wavelength->RGB).

    d^2 N / (dx dlambda) = (2 pi alpha z^2 / lambda^2) (1 - 1/(beta^2 n(lambda)^2))
"""

import numpy as np
import matplotlib.pyplot as plt

ALPHA = 1.0 / 137.035999
TWO_PI_ALPHA = 2.0 * np.pi * ALPHA


def n_water(lam_um):
    """Refractive index of water at 20 C (Daimon & Masumura 2007)."""
    L = lam_um ** 2
    n2m1 = (
        5.672526103e-1 * L / (L - 5.085550461e-3)
        + 1.736581125e-1 * L / (L - 1.814938654e-2)
        + 2.121531502e-2 * L / (L - 2.617260739e-2)
        + 1.138493213e-1 * L / (L - 1.073888649e+1)
    )
    return np.sqrt(n2m1 + 1.0)


def wavelength_to_rgb(wavelength, gamma=0.8):
    """Approximate sRGB for a visible wavelength in nm.

    Based on Dan Bruton: http://www.physics.sfasu.edu/astro/color/spectra.html
    Outside 380-750 nm returns black. Returns floats in [0, 1].
    """
    w = float(wavelength)
    if 380 <= w <= 440:
        att = 0.3 + 0.7 * (w - 380) / (440 - 380)
        R = ((-(w - 440) / (440 - 380)) * att) ** gamma
        G = 0.0
        B = (1.0 * att) ** gamma
    elif 440 <= w <= 490:
        R = 0.0
        G = ((w - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= w <= 510:
        R = 0.0
        G = 1.0
        B = (-(w - 510) / (510 - 490)) ** gamma
    elif 510 <= w <= 580:
        R = ((w - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= w <= 645:
        R = 1.0
        G = (-(w - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= w <= 750:
        att = 0.3 + 0.7 * (750 - w) / (750 - 645)
        R = (1.0 * att) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0
    return (R, G, B)


# Scenario
z = 1
beta = 1.0
medium_label = "20°C water"

# Wavelength grid (nm)
lam_nm = np.linspace(380.0, 750.0, 1000) #used to start at 350
lam_um = lam_nm * 1e-3
lam_cm = lam_nm * 1e-7
n = n_water(lam_um)

# Frank-Tamm in lambda, photons / (cm · nm)
dN_dxdlam = TWO_PI_ALPHA * z**2 / lam_cm**2 * (1.0 - 1.0 / (beta**2 * n**2)) * 1e-7
dN_dxdlam = np.where(beta * n > 1.0, dN_dxdlam, np.nan)

# Plot
fig, ax = plt.subplots(figsize=(6, 8))

# --- Visible-spectrum background via Bruton wavelength->RGB ---
N_grad = 1000
lams_grad = np.linspace(lam_nm[0], lam_nm[-1], N_grad)
rgb = np.array([wavelength_to_rgb(w) for w in lams_grad])  # (N, 3) in [0,1]
rgb_img = rgb.reshape(1, N_grad, 3)                        # (1, N, 3) for imshow

y_top = 1.85
ax.imshow(
    rgb_img,
    aspect="auto",
    extent=[lam_nm[0], lam_nm[-1], 0, y_top],
    alpha=0.40,
    zorder=0,
    interpolation="bilinear",
)

# Curve on top
ax.plot(lam_nm, dN_dxdlam, lw=2.4, color="#0b1d3a", zorder=3)

ax.set_xlabel(r"Wavelength  $\lambda$  (nm)", fontsize=12)
ax.set_ylabel(r"$\dfrac{d^2 N}{dx\,d\lambda}$   (photons cm$^{-1}$ nm$^{-1}$)",
              fontsize=12)
ax.set_title(
    f"Cherenkov spectrum in {medium_label}\n"
    rf"$n \approx 1.33$, $z={z}$, $\beta={beta:g}$  —  (Франк & Тамм 1937)",
    fontsize=12,
)
ax.set_xlim(lam_nm[0], lam_nm[-1])
ax.set_ylim(0, y_top)
ax.grid(alpha=0.3, zorder=1)

# Reference markers
for lam_ref in [405.0, 670.0]:
    n_ref = float(n_water(lam_ref * 1e-3))
    y_ref = TWO_PI_ALPHA / (lam_ref * 1e-7) ** 2 * (1 - 1/n_ref**2) * 1e-7
    ax.plot([lam_ref], [y_ref], "x", color="#d2691e", ms=5, zorder=4)
    ax.annotate(f"{y_ref:.2f}", xy=(lam_ref, y_ref),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, color="#3d2208", zorder=4)

# Integrated yield over visible range
uvEdge = 380
irEdge = 750
mask = (lam_nm >= uvEdge) & (lam_nm <= irEdge)
N_vis = np.trapezoid(dN_dxdlam[mask], lam_nm[mask])
ax.text(
    0.97, 0.95,
    f"∫({uvEdge}–{irEdge} nm) ≈ {N_vis:.0f} photons / cm",
    transform=ax.transAxes, ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.92),
    zorder=5,
)

plt.tight_layout()
print(f"Integrated {uvEdge}-{irEdge} nm: {N_vis:.1f} photons/cm")
