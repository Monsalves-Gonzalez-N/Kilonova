"""
Precomputa el cubo de magnitudes aparentes de la grid LANL de kilonovas.

Output: kn_grid_v1.npz con
    mags        : float32 (N_spec, N_t, 54, N_band, N_z)  — mag AB aparente
    times       : float32 (N_t,)                          — días desde merger
    z_bins      : float32 (N_z,)                          — z uniforme [z_min, z_max]
    spec_names  : object  (N_spec,)                       — basename del spec file
    bands       : object  (N_band,)                       — orden canónico
    angles      : int32   (54,)                           — 0..53

Ejecución: parallel sobre specs con multiprocessing.Pool.
    python build_kn_grid.py --n_workers 32 --output /Volumes/Elements/kilonova/kn_grid_v1.npz

Vectorización: para cada (spec, band, z) hace un solo np.trapezoid sobre
(N_t * 54, N_wav_masked) → ~6 * N_z = 1200 trapz por spec → ~2 s/spec en single-thread.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
import astropy.units as u

# Permite import de kilonova_syntetic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kilonova_syntetic import (  # noqa: E402
    parse_spec,
    load_roman_filters,
    C_ANG_S,
    N_ANGLE_BINS,
)

# ─── Globals para workers (set por _init_worker) ─────────────────────────────
_FILTERS: dict | None = None
_LAM_AA_REF: np.ndarray | None = None
_TIMES_REF: np.ndarray | None = None
_Z_BINS: np.ndarray | None = None
_D_L_PC: np.ndarray | None = None      # (N_z,) en pc
_BANDS: list[str] | None = None
_PRECOMP: dict | None = None            # cache de (band, z_idx) → (mask, T_interp, lam_m, scale)


def _init_worker(z_bins, lam_AA_ref, times_ref, bands):
    """Inicializa cada worker con cosmología, filtros y precomputos por (band, z)."""
    global _FILTERS, _LAM_AA_REF, _TIMES_REF, _Z_BINS, _D_L_PC, _BANDS, _PRECOMP

    _FILTERS = load_roman_filters()
    _LAM_AA_REF = lam_AA_ref
    _TIMES_REF = times_ref
    _Z_BINS = z_bins
    _BANDS = bands
    _D_L_PC = Planck18.luminosity_distance(z_bins).to(u.pc).value.astype(np.float64)

    # Precompute por (band, z): mask de longitudes de onda, T interpolado, lam_obs masked, scale
    _PRECOMP = {}
    for b in bands:
        filt = _FILTERS[b]
        filt_lam = np.asarray(filt.wavelength, dtype=np.float64)
        filt_T = np.asarray(filt.response, dtype=np.float64)
        for zi, z in enumerate(z_bins):
            lam_obs = lam_AA_ref * (1.0 + z)
            mask = (lam_obs >= filt_lam[0]) & (lam_obs <= filt_lam[-1])
            if mask.sum() < 2:
                _PRECOMP[(b, zi)] = None
                continue
            lam_m = lam_obs[mask]
            T_interp = np.interp(lam_m, filt_lam, filt_T)
            scale = (10.0 / _D_L_PC[zi]) ** 2 / (1.0 + z)
            # Pesos para trapezoid: usaremos f * T * lam y T / lam
            # Den (independiente de t y angle) puede precomputarse:
            den = float(np.trapezoid(T_interp / lam_m, lam_m))
            _PRECOMP[(b, zi)] = {
                "mask": mask,
                "T_lam": (T_interp * lam_m).astype(np.float64),  # para num
                "lam_m": lam_m,
                "scale": float(scale * N_ANGLE_BINS),  # incluye factor isotrópico
                "den": den,
            }


def _get_times_only(filepath: str) -> np.ndarray:
    """Extrae solo el array de times de un spec file (rápido, sin parsear datos)."""
    times = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") and "time[d]=" in line:
                times.append(float(line.split("time[d]=")[1]))
    return np.asarray(times, dtype=np.float64)


def _process_spec(spec_path: str) -> tuple[str, np.ndarray]:
    """
    Procesa un spec file completo. Devuelve (basename, mags_array).

    mags_array : float32 (N_t_ref, 54, N_band, N_z)
        N_t_ref es la longitud canónica. Specs más cortos quedan padeados con NaN
        en los timesteps finales.
    """
    times_d, lam_AA, flux = parse_spec(spec_path)

    # Sanity: mismo grid de wavelength que la referencia
    if not np.array_equal(lam_AA, _LAM_AA_REF):
        raise ValueError(f"lam_AA mismatch en {spec_path}")

    # Tiempo: debe ser prefix de la grid canónica
    N_t_spec = len(times_d)
    N_t_ref = len(_TIMES_REF)
    if N_t_spec > N_t_ref:
        raise ValueError(f"N_t > ref en {spec_path}: {N_t_spec} > {N_t_ref}")
    if not np.allclose(times_d, _TIMES_REF[:N_t_spec], atol=1e-6):
        raise ValueError(f"times no coinciden con prefix de ref en {spec_path}")

    N_t = N_t_spec
    N_ang = flux.shape[2]  # 54
    N_band = len(_BANDS)
    N_z = len(_Z_BINS)

    # flux shape: (N_t, N_wav, 54). Reordenamos a (N_t*N_ang, N_wav) por banda/z.
    # Output con shape canónica (N_t_ref); pad con NaN en los timesteps finales.
    out = np.full((N_t_ref, N_ang, N_band, N_z), np.nan, dtype=np.float32)

    # Pre-aplana flux a (N_t*N_ang, N_wav) una sola vez
    # flux[t, :, a] → orden t-major, angle-minor: row = t*N_ang + a
    flux_flat = np.transpose(flux, (0, 2, 1)).reshape(N_t * N_ang, -1)  # (N_t*N_ang, N_wav)

    for bi, b in enumerate(_BANDS):
        for zi in range(N_z):
            pre = _PRECOMP[(b, zi)]
            if pre is None:
                continue
            mask = pre["mask"]
            T_lam = pre["T_lam"]              # (N_wav_m,)
            lam_m = pre["lam_m"]              # (N_wav_m,)
            scale = pre["scale"]              # incluye N_ANGLE_BINS
            den = pre["den"]

            f_m = flux_flat[:, mask] * scale  # (N_t*N_ang, N_wav_m)
            num = np.trapezoid(f_m * T_lam, lam_m, axis=1)  # (N_t*N_ang,)

            with np.errstate(invalid="ignore", divide="ignore"):
                F_nu = num / den / C_ANG_S
                m_AB = -2.5 * np.log10(np.where(F_nu > 0, F_nu, np.nan)) - 48.60

            out[:N_t, :, bi, zi] = m_AB.reshape(N_t, N_ang).astype(np.float32)

    return Path(spec_path).name, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec_dir", default="/Volumes/Elements/kilonova/LANL_grid/kn_sim_cube_v1")
    p.add_argument("--output", default="/Volumes/Elements/kilonova/kn_grid_v1.npz")
    p.add_argument("--z_min", type=float, default=0.03)
    p.add_argument("--z_max", type=float, default=0.10)
    p.add_argument("--n_z", type=int, default=200)
    p.add_argument("--bands", nargs="+", default=["R", "Z", "Y", "J", "H", "F"])
    p.add_argument("--n_workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--limit", type=int, default=None,
                   help="Procesa solo los primeros N specs (para test)")
    args = p.parse_args()

    spec_files = sorted(Path(args.spec_dir).glob("*_spec_*.dat"))
    if args.limit:
        spec_files = spec_files[: args.limit]
    if not spec_files:
        raise FileNotFoundError(f"No spec files in {args.spec_dir}")

    print(f"[build_kn_grid] {len(spec_files)} spec files")
    print(f"[build_kn_grid] z_bins: linspace({args.z_min}, {args.z_max}, {args.n_z})")
    print(f"[build_kn_grid] bands: {args.bands}")
    print(f"[build_kn_grid] workers: {args.n_workers}")
    print(f"[build_kn_grid] output: {args.output}")

    z_bins = np.linspace(args.z_min, args.z_max, args.n_z, dtype=np.float64)

    # Prescan rápido: encuentra el spec con más timesteps para usar como grid canónica.
    # Solo extrae las líneas '# time[d]=' (no parsea datos), así que es ~100x más rápido
    # que parse_spec completo.
    print(f"[build_kn_grid] prescanning {len(spec_files)} specs for canonical time grid...")
    t_pre = time.time()
    from multiprocessing import Pool as _Pool
    with _Pool(args.n_workers) as _pool:
        all_times = _pool.map(_get_times_only, [str(s) for s in spec_files])
    n_t_per_spec = np.array([len(t) for t in all_times])
    longest_idx = int(np.argmax(n_t_per_spec))
    print(f"[build_kn_grid]   prescan done in {time.time() - t_pre:.1f}s")
    print(f"[build_kn_grid]   N_t range: min={n_t_per_spec.min()}, max={n_t_per_spec.max()}")
    print(f"[build_kn_grid]   canonical spec: {spec_files[longest_idx].name} (N_t={n_t_per_spec.max()})")

    # Verifica consistencia: todos los specs deben ser prefix de la grid canónica
    canonical_times = all_times[longest_idx]
    bad = []
    for i, t in enumerate(all_times):
        if not np.allclose(t, canonical_times[: len(t)], atol=1e-6):
            bad.append(spec_files[i].name)
    if bad:
        print(f"[build_kn_grid] WARNING: {len(bad)} specs no son prefix de la canónica:")
        for n in bad[:5]:
            print(f"    {n}")
        raise ValueError("grid temporal inconsistente; revisar manualmente")

    # Parse del spec canónico para obtener lam_AA
    print(f"[build_kn_grid] parsing canonical spec for lam_AA...")
    times_ref, lam_AA_ref, _ = parse_spec(str(spec_files[longest_idx]))
    print(f"[build_kn_grid]   N_t={len(times_ref)}  N_wav={len(lam_AA_ref)}")

    N_spec = len(spec_files)
    N_t = len(times_ref)
    N_ang = N_ANGLE_BINS
    N_band = len(args.bands)
    N_z = args.n_z

    cube_bytes = N_spec * N_t * N_ang * N_band * N_z * 4
    print(f"[build_kn_grid] cube size: ({N_spec}, {N_t}, {N_ang}, {N_band}, {N_z}) "
          f"= {cube_bytes / 1e9:.2f} GB float32")

    # Reservar el cubo final
    mags_cube = np.full((N_spec, N_t, N_ang, N_band, N_z), np.nan, dtype=np.float32)
    spec_names = np.empty(N_spec, dtype=object)

    # Map basename → idx (orden alfabético = orden de spec_files)
    name_to_idx = {f.name: i for i, f in enumerate(spec_files)}

    t_start = time.time()

    if args.n_workers == 1:
        # Modo serial (debug)
        _init_worker(z_bins, lam_AA_ref, times_ref, args.bands)
        for i, sf in enumerate(spec_files):
            name, m = _process_spec(str(sf))
            mags_cube[name_to_idx[name]] = m
            spec_names[name_to_idx[name]] = name
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_start
                eta = elapsed / (i + 1) * (N_spec - i - 1)
                print(f"  [{i+1}/{N_spec}] {name}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    else:
        from multiprocessing import Pool
        with Pool(
            processes=args.n_workers,
            initializer=_init_worker,
            initargs=(z_bins, lam_AA_ref, times_ref, args.bands),
        ) as pool:
            for i, (name, m) in enumerate(
                pool.imap_unordered(_process_spec, [str(s) for s in spec_files], chunksize=2)
            ):
                idx = name_to_idx[name]
                mags_cube[idx] = m
                spec_names[idx] = name
                if (i + 1) % 20 == 0 or i == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / (i + 1) * (N_spec - i - 1)
                    print(f"  [{i+1}/{N_spec}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"[build_kn_grid] done in {time.time() - t_start:.0f}s")
    print(f"[build_kn_grid] saving → {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mags=mags_cube,
        times=times_ref.astype(np.float32),
        z_bins=z_bins.astype(np.float32),
        spec_names=spec_names,
        bands=np.array(args.bands, dtype=object),
        angles=np.arange(N_ang, dtype=np.int32),
    )

    n_finite = int(np.isfinite(mags_cube).sum())
    n_total = mags_cube.size
    print(f"[build_kn_grid] finite mags: {n_finite}/{n_total} ({100*n_finite/n_total:.1f}%)")
    print(f"[build_kn_grid] file size: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
