"""
Fotometría sintética de kilonovas a partir de los archivos _spec_ de la grid LANL
(Wollaeger+2021, Zenodo 7335961).

Uso típico:
    from kilonova_syntetic import parse_spec, apparent_abmag, apparent_lightcurve, FILTERS

    times, lam_AA, flux = parse_spec(spec_file)
    m_J = apparent_abmag(lam_AA, flux[t_idx, :, angle_idx], FILTERS["J"], z=0.05)

Dos casos de uso independientes:

1. validate_filters(): compara nuestras magnitudes AB sintéticas (z=0, R=10pc)
   contra los valores _mags_ pre-computados de LANL. La única diferencia esperada
   es la elección de filtro (e.g. 2MASS Ks vs MOSFIRE Ks).

2. apparent_lightcurve(): coloca un espectro rest-frame a redshift z arbitrario
   usando cosmología Planck18 y calcula magnitudes AB aparentes en JHKs.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.constants as const
import astropy.units as u
import speclite.filters as sf
from pathlib import Path

# ── Constants from astropy ─────────────────────────────────────────────────────
C_ANG_S     = const.c.to(u.AA / u.s).value    # speed of light [Å/s]
CM_TO_ANG   = (1 * u.cm).to(u.AA).value        # 1 cm → Å
Z_AT2017    = 0.009783                          # redshift NGC 4993
N_ANGLE_BINS = 54   # LANL spec flux is per angular bin; multiply by N_ANGLE_BINS
                    # to recover the isotropic-equivalent flux seen by an observer
                    # (same correction stated in data_format.pdf for luminosities)

# ── Filter loading ─────────────────────────────────────────────────────────────
FILTERS = {
    "J":  sf.load_filter("twomass-J"),
    "H":  sf.load_filter("twomass-H"),
    "Ks": sf.load_filter("twomass-Ks"),
}


def load_roman_filters(kcor_fits=None, group_name="roman"):
    """
    Carga las curvas de transmisión Roman desde el HDU `FilterTrans` de un
    archivo kcor SNANA y las registra como filtros speclite.

    Parameters
    ----------
    kcor_fits  : path al kcor_ROMAN.fits. Si None, usa la ruta del repo
                 hourglass_snana_sims/kcor/kcor_ROMAN.fits relativa a este script.
    group_name : nombre de grupo speclite (filtros quedan como "roman-R", etc.).

    Returns
    -------
    dict[str, speclite.filters.FilterResponse]  — claves R, Z, Y, J, H, F, W.
    """
    from astropy.io import fits
    from speclite.filters import FilterResponse

    if kcor_fits is None:
        kcor_fits = Path(__file__).resolve().parent / "kcor" / "kcor_ROMAN.fits"

    with fits.open(kcor_fits) as hdul:
        tbl = hdul["FilterTrans"].data
        wave = np.asarray(tbl["wavelength (A)"], dtype=float)
        cols = {
            "R": "R062-R", "Z": "Z087-Z", "Y": "Y106-Y",
            "J": "J129-J", "H": "H158-H", "F": "F184-F", "W": "W146-W",
        }
        out = {}
        for short, col in cols.items():
            resp = np.asarray(tbl[col], dtype=float)
            # speclite exige extremos en 0 y respuesta no negativa
            resp = np.clip(resp, 0.0, None)
            resp[0] = 0.0
            resp[-1] = 0.0
            out[short] = FilterResponse(
                wavelength=wave * u.AA,
                response=resp,
                meta=dict(group_name=group_name, band_name=short),
            )
    return out

# ── Cosmology helper ───────────────────────────────────────────────────────────
def luminosity_distance_pc(z):
    """Luminosity distance in pc for redshift z using Planck18."""
    return Planck18.luminosity_distance(z).to(u.pc).value

def distance_modulus(z):
    """Distance modulus μ = 5 log10(D_L / 10 pc) using Planck18."""
    return 5.0 * np.log10(luminosity_distance_pc(z) / 10.0)


# ── Spectrum parser ────────────────────────────────────────────────────────────
def parse_spec(filepath):
    """
    Parse a LANL _spec_ file.

    Returns
    -------
    times  : (N_t,)            time [days since merger]
    lam_AA : (N_wav,)          rest-frame central wavelengths [Å]
    flux   : (N_t, N_wav, 54)  F_λ [erg/s/Å/cm²] at R=10 pc
    """
    times, spectra, current = [], [], []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if current:
                    spectra.append(np.array(current))
                    current = []
                times.append(float(line.split("time[d]=")[1]))
            elif line:
                current.append([float(x) for x in line.split()])

    if current:
        spectra.append(np.array(current))

    spectra = np.array(spectra)                        # (N_t, N_wav, 56)
    lam_lo  = spectra[0, :, 0] * CM_TO_ANG            # Å
    lam_hi  = spectra[0, :, 1] * CM_TO_ANG            # Å
    lam_AA  = 0.5 * (lam_lo + lam_hi)                 # central wavelength [Å]
    flux    = spectra[:, :, 2:]                        # (N_t, N_wav, 54)

    return np.array(times), lam_AA, flux


# ── Mags file parser ───────────────────────────────────────────────────────────
BAND_BLOCK = {"r": 1, "i": 2, "z": 3, "y": 4, "J": 5, "H": 6, "K": 7, "S": 8}

def parse_mags(filepath, band):
    """
    Parse one band block from a LANL _mags_ file.
    Values are absolute AB magnitudes at R=10 pc (rest frame, z=0).

    Returns
    -------
    times : (N_t,)      time [days]
    mags  : (N_t, 54)   AB magnitude per angular bin
    """
    with open(filepath) as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    rows   = []
    for line in blocks[BAND_BLOCK[band]].split("\n"):
        if line.startswith("#"):
            continue
        vals = line.split()
        if len(vals) >= 56:
            rows.append([float(v) for v in vals])

    rows  = np.array(rows)
    return rows[:, 1], rows[:, 2:]   # times [days], mags (N_t, 54)


# ── Core photometry: rest-frame spectrum → AB magnitude ───────────────────────
def restframe_abmag(lam_AA, f_lam, filt):
    """
    Compute absolute AB magnitude at R=10 pc from a rest-frame spectrum.
    No redshift applied — pure filter convolution.

    Parameters
    ----------
    lam_AA : (N,) rest-frame wavelengths [Å]
    f_lam  : (N,) F_λ [erg/s/Å/cm²] at R=10 pc
    filt   : speclite Filter object

    Returns
    -------
    M_AB : float  (absolute AB magnitude)
    """
    filt_lam = filt.wavelength   # Å
    filt_T   = filt.response

    mask = (lam_AA >= filt_lam[0]) & (lam_AA <= filt_lam[-1])
    if mask.sum() < 2:
        return np.nan

    T_interp = np.interp(lam_AA[mask], filt_lam, filt_T)
    lam_m    = lam_AA[mask]
    f_m      = f_lam[mask] * N_ANGLE_BINS   # flux per bin → observer flux

    # AB: <F_ν> = (1/c) · ∫ F_λ·T·λ dλ / ∫ T/λ dλ
    num  = np.trapezoid(f_m * T_interp * lam_m, lam_m)
    den  = np.trapezoid(T_interp / lam_m,        lam_m)
    if den <= 0 or num <= 0:
        return np.nan

    F_nu = num / den / C_ANG_S      # erg/s/Hz/cm²
    return -2.5 * np.log10(F_nu) - 48.60


# ── Production: place spectrum at redshift z → apparent magnitude ──────────────
def apparent_abmag(lam_AA, f_lam, filt, z):
    """
    Compute apparent AB magnitude for a source at redshift z.

    The rest-frame spectrum (at R=10 pc) is redshifted and scaled to the
    luminosity distance from Planck18 cosmology.

    Parameters
    ----------
    lam_AA : (N,) rest-frame wavelengths [Å]
    f_lam  : (N,) F_λ [erg/s/Å/cm²] at R=10 pc
    filt   : speclite Filter object
    z      : float  redshift

    Returns
    -------
    m_AB : float  (apparent AB magnitude)
    """
    D_L_pc = luminosity_distance_pc(z)

    # Redshift wavelength grid and scale flux to observed frame
    lam_obs = lam_AA * (1.0 + z)
    scale   = (10.0 / D_L_pc)**2 / (1.0 + z)
    f_obs   = f_lam * scale

    # Convolve with filter at observer-frame wavelengths
    filt_lam = filt.wavelength
    filt_T   = filt.response

    mask = (lam_obs >= filt_lam[0]) & (lam_obs <= filt_lam[-1])
    if mask.sum() < 2:
        return np.nan

    T_interp = np.interp(lam_obs[mask], filt_lam, filt_T)
    lam_m    = lam_obs[mask]
    f_m      = f_obs[mask] * N_ANGLE_BINS   # flux per bin → observer flux

    num  = np.trapezoid(f_m * T_interp * lam_m, lam_m)
    den  = np.trapezoid(T_interp / lam_m,        lam_m)
    if den <= 0 or num <= 0:
        return np.nan

    F_nu = num / den / C_ANG_S
    return -2.5 * np.log10(F_nu) - 48.60


# ── Roman DEEP observational noise ────────────────────────────────────────────
ZP_SNANA = 27.5

# Parámetros Roman (de make_roman_simlib.py).
# DEEP cubre Y/J/H/F; WIDE cubre R/Z/Y/J. NEA se obtiene del parquet de Hourglass.
# Valores tomados directamente del simlib rose21_25_percent.simlib (LIBID 1 WIDE,
# LIBID 69 DEEP). Validados contra hourglass_photometry.parquet con test
# fila-a-fila usando los params del propio parquet (psf_nea, sky_sig, read_noise,
# zp): mediana global sigma_pred/sigma_real = 0.986, p95 ≤ 1.02 en todas las
# bandas. El ~1.4% residual y la cola en p05 (≈0.65 en F/H DEEP) se deben al
# ruido de la galaxia host, que SNANA suma como término extra y no podemos
# reproducir sin un catálogo de hosts. Aceptable para simulación de KN sin host.
ROMAN_DEEP = {
    "Y": {"zp": 32.549, "nea":  7.895, "sky_sig": 10.392, "read_noise": 7.450},
    "J": {"zp": 32.547, "nea":  9.210, "sky_sig": 10.421, "read_noise": 7.450},
    "H": {"zp": 32.570, "nea": 11.140, "sky_sig": 10.775, "read_noise": 7.450},
    "F": {"zp": 33.299, "nea": 16.335, "sky_sig": 17.723, "read_noise": 5.942},
}
ROMAN_WIDE = {
    "R": {"zp": 32.129, "nea":  5.575, "sky_sig":  7.133, "read_noise":  9.011},
    "Z": {"zp": 31.303, "nea":  6.695, "sky_sig":  5.648, "read_noise": 10.624},
    "Y": {"zp": 31.356, "nea":  7.895, "sky_sig":  6.000, "read_noise": 10.624},
    "J": {"zp": 31.354, "nea":  9.210, "sky_sig":  6.017, "read_noise": 10.624},
}
ROMAN_PARAMS = {"DEEP": ROMAN_DEEP, "WIDE": ROMAN_WIDE}


def apply_roman_noise(mag, band, survey="DEEP", nea=None, n_realizations=1, gain=1.0, rng=None):
    """
    Aplica ruido observacional Roman a una magnitud AB sin ruido.

    Parameters
    ----------
    mag            : float or array  — magnitud AB (output de apparent_abmag)
    band           : str — DEEP: Y,J,H,F  |  WIDE: R,Z,Y,J
    survey         : "DEEP" o "WIDE"
    nea            : float, optional — override del NEA [píxeles]. Si None,
                     usa la mediana de hourglass_photometry guardada en ROMAN_*.
    n_realizations : int   — cuántas realizaciones de ruido generar
    gain           : float — ganancia e-/ADU
    rng            : np.random.Generator, optional

    Returns
    -------
    fluxcal_obs : (n_realizations, N)  — flujo con ruido [ZP=27.5]
    sigma       : (N,)                 — sigma del ruido (sin fluctuaciones)
    mag_obs     : (n_realizations, N)  — mag con ruido (nan si no detección)
    """
    rng = rng or np.random.default_rng()
    p = ROMAN_PARAMS[survey][band]
    if nea is None:
        nea = p["nea"]

    mag = np.asarray(mag)
    fluxcal_true = 10 ** ((ZP_SNANA - mag) / 2.5)

    # Escala ADU nativos → fluxcal
    scale = 10 ** ((ZP_SNANA - p["zp"]) / 2.5)

    # Varianza total en unidades fluxcal
    # SNANA simlib: σ²_ADU = NEA·(skysig² + (RN/g)²) + F_ADU/g, luego ×scale²
    # → en fluxcal: el término de fuente queda con UN solo factor scale.
    var = (
        scale * fluxcal_true / gain                                         # Poisson fuente
        + nea * (p["sky_sig"]**2 + (p["read_noise"] / gain)**2) * scale**2  # fondo + lectura
    )
    sigma = np.sqrt(var)

    noise = rng.normal(0, sigma, size=(n_realizations, mag.size))
    fluxcal_obs = fluxcal_true + noise

    with np.errstate(invalid="ignore", divide="ignore"):
        mag_obs = np.where(
            fluxcal_obs > 0,
            ZP_SNANA - 2.5 * np.log10(np.maximum(fluxcal_obs, 1e-30)),
            np.nan,
        )

    return fluxcal_obs, sigma, mag_obs


# ── Use case 1: filter validation (z=0, R=10pc) ───────────────────────────────
def validate_filters(spec_file, mags_file, angle_idx=0):
    """
    Compare rest-frame synthetic magnitudes (our filters) vs LANL _mags_.
    Both at z=0, R=10pc — any residual is purely due to filter differences.
    """
    print(f"\n[VALIDATION] angle bin {angle_idx}")
    print(f"  Spec: {Path(spec_file).name}")
    print(f"  Mags: {Path(mags_file).name}")

    times_s, lam_AA, flux = parse_spec(spec_file)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Filter validation (z=0, R=10pc) — angle bin {angle_idx}\n"
        f"{Path(spec_file).stem.replace('_spec_2020-05-24', '')}",
        fontsize=9
    )

    for ax, (band, filt) in zip(axes, FILTERS.items()):
        synth = np.array([
            restframe_abmag(lam_AA, flux[i, :, angle_idx], filt)
            for i in range(len(times_s))
        ])

        mags_band = "K" if band == "Ks" else band
        times_m, mags_m = parse_mags(mags_file, mags_band)
        paper = mags_m[:, angle_idx]

        rms = np.nanstd(synth - np.interp(times_s, times_m, paper))

        ax.plot(times_s, synth,       label="Synthetic 2MASS (this work)", lw=1.5)
        ax.plot(times_m, paper, "--", label="LANL _mags_ (their filters)", lw=1.5)
        ax.invert_yaxis()
        ax.set_ylabel(f"M_AB {band} [R=10pc]", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        suffix = " — 2MASS Ks vs MOSFIRE Ks" if band == "Ks" else ""
        ax.set_title(f"RMS = {rms:.4f} mag{suffix}", fontsize=9)

    axes[-1].set_xlabel("Time [days since merger]", fontsize=10)
    plt.tight_layout()
    out = f"validation_filters_angle{angle_idx}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")


# ── Use case 2: apparent light curve at arbitrary redshift ────────────────────
def apparent_lightcurve(spec_file, angle_idx=0, redshifts=(0.009783,)):
    """
    Compute and plot apparent JHKs light curves at one or more redshifts.

    Parameters
    ----------
    redshifts : tuple of floats — e.g. (0.01, 0.05, 0.1, 0.3)
    """
    print(f"\n[APPARENT LC] angle bin {angle_idx}  z = {redshifts}")

    times_s, lam_AA, flux = parse_spec(spec_file)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Apparent light curves — angle bin {angle_idx}\n"
        f"{Path(spec_file).stem.replace('_spec_2020-05-24', '')}",
        fontsize=9
    )

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(redshifts)))

    for ax, (band, filt) in zip(axes, FILTERS.items()):
        for z, col in zip(redshifts, colors):
            mu = distance_modulus(z)
            D_L_pc = luminosity_distance_pc(z)
            mags_z = np.array([
                apparent_abmag(lam_AA, flux[i, :, angle_idx], filt, z)
                for i in range(len(times_s))
            ])
            ax.plot(times_s, mags_z, color=col, lw=1.5,
                    label=f"z={z:.4f}  μ={mu:.2f}  D_L={D_L_pc/1e6:.1f} Mpc")

        ax.invert_yaxis()
        ax.set_ylabel(f"m_AB {band}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time [days since merger]", fontsize=10)
    plt.tight_layout()
    out = f"apparent_lc_angle{angle_idx}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"  Saved: {out}")
