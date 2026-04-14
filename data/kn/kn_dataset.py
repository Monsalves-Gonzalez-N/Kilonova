"""
KilonovaContrastiveDataset — dataset contrastivo de curvas de luz KN
para contrastive learning, compatible con el pipeline de funciones.py.

Lee una grid pre-computada de magnitudes aparentes (mags.npy) en vez de
correr parse_spec + apparent_abmag on-the-fly.

Grid pre-computada (kn_grid_v1/):
    mags.npy       — (N_spec, N_t, 54, N_band, N_z)  magnitudes AB
    times.npy      — (N_t,)  días desde merger
    z_bins.npy     — (N_z,)  valores de redshift
    spec_names.npy — (N_spec,)  nombres de los modelos
    bands.npy      — (N_band,)  nombres de bandas ['R','Z','Y','J','H','F']

Token de 6 bandas (ALL_BANDS = R, Z, Y, J, H, F):
    Cada KN se simula en modo WIDE (R,Z,Y,J) o DEEP (Y,J,H,F), elegido
    aleatoriamente. Las 2 bandas del otro modo quedan con obs_flag=0.
    Esto es consistente con la restricción del survey Roman: en una noche
    se observa en WIDE o DEEP, nunca ambos.

Uso típico:
    from kn_dataset import KilonovaContrastiveDataset
    from funciones import collate_contrastive, worker_init_fn, ALL_BANDS
    from torch.utils.data import DataLoader

    survey_mjds = {b: np.sort(phot['mjd'][phot['band']==b].unique()) for b in ALL_BANDS}

    ds = KilonovaContrastiveDataset(
        grid_dir="/Volumes/Elements/kilonova/kn_grid_v1",
        survey_mjds=survey_mjds,
        n_virtual=100_000,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=collate_contrastive,
                        num_workers=4, worker_init_fn=worker_init_fn)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Importa el pipeline de tokenizado/augmentación de funciones.py
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "sn" / "hourglass_snana_zenodo"))
from funciones import (  # noqa: E402
    _lc_np_to_epoch_tensor,
    _EPOCH_DAYS,
    _FEW_EPOCH_THRESHOLD,
    _count_epochs,
    _perturb_within_error,
    ALL_BANDS,
    DEEP_BANDS,
    WIDE_BANDS,
)

sys.path.insert(0, str(_THIS_DIR))
from kilonova_syntetic import apply_roman_noise, ZP_SNANA  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KilonovaContrastiveDataset(Dataset):
    """
    Dataset contrastivo para kilonovas simuladas — inyectadas en MJDs reales
    del survey, con token de 6 bandas.

    Cada KN se simula en modo WIDE o DEEP (aleatorio por sample).
    Las 4 bandas del modo elegido tienen observaciones; las 2 restantes
    quedan con obs_flag=0 en el token.

    Parameters
    ----------
    grid_dir : str | Path
        Directorio con mags.npy, times.npy, z_bins.npy, spec_names.npy, bands.npy.
    survey_mjds : dict[str, np.ndarray]
        MJDs únicos del survey por banda (las 6 bandas).
    t_max : float
        Fase máxima post-explosión en días.
    max_mag_err : float
        Cut por fila: descarta observaciones con mag_err >= max_mag_err.
    n_virtual : int
        Tamaño "virtual" del dataset.
    label : int
        Clase entera para KN (default 0).
    epoch_days : float
        Ventana temporal para binning de épocas (mismo que funciones.py).
    z_dropout : float
        Prob. de enmascarar z=0 durante training.
    seed : int
    """

    # Mismos parámetros de ventana que ContrastiveLCDataset (SN)
    CADENCE = 5.0
    N_VISITS_RANGE = (2, 7)
    MAX_MAG_ERR = 2.0

    def __init__(
        self,
        grid_dir: str | Path,
        survey_mjds: dict[str, np.ndarray],
        spec_indices: np.ndarray | None = None,
        t_max: float = 60.0,
        max_mag_err: float = 2.0,
        n_virtual: int | None = None,
        deterministic: bool = False,
        label: int = 0,
        epoch_days: float = _EPOCH_DAYS,
        z_dropout: float = 0.2,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        deterministic : bool
            Si True (val/test): idx mapea a (spec, angle) fijo.
            len = n_spec × n_angles. z y survey_mode se sampean pero
            con seed determinista por idx → reproducible.
            Si False (train): idx samplea aleatoriamente, len = n_virtual.
        n_virtual : int | None
            Solo usado si deterministic=False. Default 100_000.
        """
        grid_dir = Path(grid_dir)
        self.bands         = list(ALL_BANDS)   # siempre 6 bandas
        self.t_max         = t_max
        self.max_mag_err   = max_mag_err
        self.deterministic = deterministic
        self.label         = int(label)
        self.epoch_days    = epoch_days
        self.z_dropout     = z_dropout

        # MJDs reales del survey por banda (sorted)
        self.survey_mjds = {
            b: np.sort(np.asarray(survey_mjds[b], dtype=np.float64))
            for b in self.bands
        }

        # Rango válido para t_explosion
        all_mjds = np.concatenate(list(self.survey_mjds.values()))
        self._mjd_min = float(all_mjds.min())
        self._mjd_max = float(all_mjds.max())

        # Carga la grid pre-computada (mmap para mags — evita cargar todo en RAM)
        self.mags       = np.load(grid_dir / "mags.npy", mmap_mode="r")
        self.times      = np.load(grid_dir / "times.npy")
        self.z_bins     = np.load(grid_dir / "z_bins.npy")
        self.spec_names = np.load(grid_dir / "spec_names.npy", allow_pickle=True)
        self.grid_bands = np.load(grid_dir / "bands.npy", allow_pickle=True)

        # Índices de las 6 bandas dentro de la grid
        self.band_indices = []
        for b in self.bands:
            idx = int(np.where(self.grid_bands == b)[0][0])
            self.band_indices.append(idx)

        # Índices locales (0-5) de las bandas DEEP y WIDE dentro de ALL_BANDS
        self._deep_local = [self.bands.index(b) for b in DEEP_BANDS]
        self._wide_local = [self.bands.index(b) for b in WIDE_BANDS]

        self.n_angles = self.mags.shape[2]  # 54
        self.n_z      = self.mags.shape[4]

        # Subset de modelos (para train/val/test split por modelo)
        if spec_indices is not None:
            self.spec_indices = np.asarray(spec_indices)
        else:
            self.spec_indices = np.arange(self.mags.shape[0])
        self.n_spec = len(self.spec_indices)

        if deterministic:
            # Val/test: iterar sobre todas las combinaciones (spec, angle)
            self.n_virtual = self.n_spec * self.n_angles
        else:
            self.n_virtual = int(n_virtual) if n_virtual is not None else 100_000

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_virtual

    # ─────────────────────────────────────────────────────────────────────
    # Simulación de una KN inyectada en MJDs del survey
    # ─────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        i_spec: int,
        i_angle: int,
        i_z: int,
        t_explosion: float,
        survey_mode: str,
        rng: np.random.Generator,
    ) -> dict:
        """
        Genera un lc_np dict inyectando la KN en los MJDs reales del survey.

        Solo simula las 4 bandas del survey_mode ("DEEP" o "WIDE").
        Las 2 bandas restantes no aparecen → obs_flag=0 en el token.

        Parameters
        ----------
        t_explosion : float
            MJD de la explosión.
        survey_mode : "DEEP" | "WIDE"
            Determina qué 4 bandas se simulan y el modelo de ruido.

        Returns dict con keys: mjd, band_idx, mag, err, ul, t0
        """
        active_local = self._deep_local if survey_mode == "DEEP" else self._wide_local

        all_mjd, all_bidx, all_mag, all_err, all_ul = [], [], [], [], []

        for local_bi in active_local:
            grid_bi = self.band_indices[local_bi]
            b = self.bands[local_bi]
            band_mjds = self.survey_mjds[b]

            # Fase de cada MJD del survey relativa a la explosión
            phase = band_mjds - t_explosion
            valid = (phase > 0) & (phase <= self.t_max)
            visit_mjds   = band_mjds[valid]
            visit_phases = phase[valid]

            if len(visit_phases) == 0:
                continue

            # Curva limpia desde la grid (mmap read)
            m_dense = self.mags[i_spec, :, i_angle, grid_bi, i_z]
            finite_m = np.isfinite(m_dense)
            if finite_m.sum() < 2:
                continue
            t_model = self.times[finite_m]
            m_model = np.array(m_dense[finite_m])  # copy solo lo finito

            # Solo fases dentro del rango cubierto por el modelo
            in_range = (visit_phases >= t_model.min()) & (visit_phases <= t_model.max())
            vp = visit_phases[in_range]
            vm = visit_mjds[in_range]
            if len(vp) == 0:
                continue

            mags_visit = np.interp(vp, t_model, m_model)

            # Ruido Roman (usa el survey_mode correcto)
            _, sigma_v, mag_obs = apply_roman_noise(
                mags_visit, band=b, survey=survey_mode,
                n_realizations=1, rng=rng,
            )
            mag_obs = mag_obs[0]
            F_true = 10 ** ((ZP_SNANA - mags_visit) / 2.5)
            with np.errstate(invalid="ignore", divide="ignore"):
                snr_v   = np.where(sigma_v > 0, F_true / sigma_v, np.nan)
                mag_err = 1.0857 / snr_v

            keep = np.isfinite(mag_err) & (mag_err < self.max_mag_err) & np.isfinite(mag_obs)
            if not keep.any():
                continue

            n_keep = keep.sum()
            all_mjd.append(vm[keep])
            # band_idx en el espacio de 6 bandas (ALL_BANDS)
            all_bidx.append(np.full(n_keep, local_bi, dtype=np.int64))
            all_mag.append(mag_obs[keep].astype(np.float32))
            all_err.append(mag_err[keep].astype(np.float32))
            all_ul.append(np.zeros(n_keep, dtype=np.float32))

        if not all_mjd:
            return {"mjd": np.empty(0), "band_idx": np.empty(0, dtype=np.int64),
                    "mag": np.empty(0, np.float32), "err": np.empty(0, np.float32),
                    "ul": np.empty(0, np.float32), "t0": 0.0}

        mjd = np.concatenate(all_mjd)
        order = np.argsort(mjd)
        mjd_sorted = mjd[order]

        return {
            "mjd":      mjd_sorted,
            "band_idx": np.concatenate(all_bidx)[order],
            "mag":      np.concatenate(all_mag)[order],
            "err":      np.concatenate(all_err)[order],
            "ul":       np.concatenate(all_ul)[order],
            "t0":       float(mjd_sorted[0]),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Sampling de parámetros
    # ─────────────────────────────────────────────────────────────────────

    def _sample_physics(self, rng: np.random.Generator) -> dict:
        """Samplea la KN física (compartida entre ambas vistas)."""
        return dict(
            i_spec  = int(rng.choice(self.spec_indices)),
            i_angle = int(rng.integers(0, self.n_angles)),
            i_z     = int(rng.integers(0, self.n_z)),
        )

    def _sample_t_explosion(self, rng: np.random.Generator) -> float:
        """
        Samplea un MJD de explosión dentro del survey.
        """
        buffer = self.N_VISITS_RANGE[0] * self.CADENCE
        latest = self._mjd_max - buffer
        return float(rng.uniform(self._mjd_min, latest))

    def _window_view(self, lc_np: dict, n_visits: int, rng: np.random.Generator) -> dict:
        """
        Recorta una simulación a una ventana de n_visits × CADENCE días,
        igual que ContrastiveLCDataset._window_view en el dataset SN.
        """
        mjd = lc_np["mjd"]
        if len(mjd) == 0:
            return lc_np

        window_days = n_visits * self.CADENCE
        mjd_min = mjd[0]
        mjd_max = mjd[-1]
        latest_start = mjd_max - window_days

        if latest_start < mjd_min:
            latest_start = mjd_min

        start = float(rng.uniform(mjd_min, latest_start + self.CADENCE))
        end   = start + window_days

        mask = (mjd >= start) & (mjd < end) & (lc_np["err"] < self.MAX_MAG_ERR)
        return {
            "mjd":      mjd[mask],
            "band_idx": lc_np["band_idx"][mask],
            "mag":      lc_np["mag"][mask],
            "err":      lc_np["err"][mask],
            "ul":       lc_np["ul"][mask],
            "t0":       float(mjd[mask][0]) if mask.any() else lc_np["t0"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # __getitem__: dos vistas contrastivas
    # ─────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        if self.deterministic:
            # idx → (spec, angle) fijo; z, t_explosion, survey_mode con seed por idx
            rng = np.random.default_rng(idx)
            i_local = idx // self.n_angles
            i_angle = idx % self.n_angles
            i_spec  = int(self.spec_indices[i_local])
            i_z     = int(rng.integers(0, self.n_z))
            physics = dict(i_spec=i_spec, i_angle=i_angle, i_z=i_z)
        else:
            rng = np.random.default_rng(self._rng.integers(0, 2**63 - 1))
            physics = None  # se samplea abajo

        # n_visits compartido
        n_visits = int(rng.integers(self.N_VISITS_RANGE[0],
                                    self.N_VISITS_RANGE[1] + 1))

        # Survey mode aleatorio pero compartido entre ambas vistas
        survey_mode = "DEEP" if rng.random() < 0.5 else "WIDE"

        for _ in range(8):
            if physics is None:
                physics = self._sample_physics(rng)

            t_exp_1 = self._sample_t_explosion(rng)
            lc1_full = self.simulate(**physics, t_explosion=t_exp_1,
                                     survey_mode=survey_mode, rng=rng)
            lc1 = self._window_view(lc1_full, n_visits, rng)

            t_exp_2 = self._sample_t_explosion(rng)
            lc2_full = self.simulate(**physics, t_explosion=t_exp_2,
                                     survey_mode=survey_mode, rng=rng)
            lc2 = self._window_view(lc2_full, n_visits, rng)

            if len(lc1["mjd"]) >= 2 and len(lc2["mjd"]) >= 2:
                break

            if self.deterministic:
                # No re-samplear physics, solo reintentar t_explosion
                pass
            else:
                physics = None  # re-samplear en siguiente intento
        else:
            lc1_full = self.simulate(**physics, t_explosion=self._mjd_min,
                                     survey_mode=survey_mode, rng=rng)
            lc2_full = self.simulate(**physics, t_explosion=self._mjd_min,
                                     survey_mode=survey_mode, rng=rng)
            lc1 = self._window_view(lc1_full, self.N_VISITS_RANGE[1], rng)
            lc2 = self._window_view(lc2_full, self.N_VISITS_RANGE[1], rng)

        # Few-epoch: activar solo si la simulación completa tiene pocas epochs
        # (evento intrínsecamente corto). No activar si la curva completa es
        # larga y solo la view cayó en pocas epochs por mala suerte del window.
        n_ep_full1 = _count_epochs(lc1_full, self.epoch_days)
        n_ep_full2 = _count_epochs(lc2_full, self.epoch_days)
        if n_ep_full1 <= _FEW_EPOCH_THRESHOLD or n_ep_full2 <= _FEW_EPOCH_THRESHOLD:
            base = lc1_full if len(lc1_full["mjd"]) >= len(lc2_full["mjd"]) else lc2_full
            lc1 = _perturb_within_error(base, rng)
            lc2 = _perturb_within_error(base, rng)

        z_true = float(self.z_bins[physics["i_z"]])

        x1 = _lc_np_to_epoch_tensor(lc1, len(self.bands), self.epoch_days)
        x2 = _lc_np_to_epoch_tensor(lc2, len(self.bands), self.epoch_days)

        # Guard: si ambas vistas están vacías, crear token mínimo (1 época, todo ceros)
        # para evitar tensores (0, token_dim) que rompen collate/loss.
        token_dim = 1 + 4 * len(self.bands)
        if x1.shape[0] == 0:
            x1 = torch.zeros(1, token_dim)
        if x2.shape[0] == 0:
            x2 = torch.zeros(1, token_dim)

        z = 0.0 if (self.z_dropout > 0 and rng.random() < self.z_dropout) else z_true
        return x1, x2, float(z), self.label

    # ─────────────────────────────────────────────────────────────────────
    # Visualización
    # ─────────────────────────────────────────────────────────────────────

    def plot(self, physics: dict | None = None, survey_mode: str | None = None,
             seed: int | None = None):
        """
        Plotea una KN simulada: curva completa + dos vistas con ventana.
        """
        import matplotlib.pyplot as plt
        from funciones import _band_color, BAND_LAMBDA_NM

        rng = np.random.default_rng(seed)
        if physics is None:
            physics = self._sample_physics(rng)
        if survey_mode is None:
            survey_mode = "DEEP" if rng.random() < 0.5 else "WIDE"

        n_visits = int(rng.integers(self.N_VISITS_RANGE[0],
                                    self.N_VISITS_RANGE[1] + 1))

        # Dos simulaciones independientes (distinto t_explosion, mismo modelo)
        # — refleja lo que hace __getitem__: misma física, distinta "suerte"
        # de cuándo explota respecto al survey.
        t_exp_1 = self._sample_t_explosion(rng)
        lc_full1 = self.simulate(**physics, t_explosion=t_exp_1,
                                  survey_mode=survey_mode, rng=rng)
        lc_v1 = self._window_view(lc_full1, n_visits, rng)

        t_exp_2 = self._sample_t_explosion(rng)
        lc_full2 = self.simulate(**physics, t_explosion=t_exp_2,
                                  survey_mode=survey_mode, rng=rng)
        lc_v2 = self._window_view(lc_full2, n_visits, rng)

        if len(lc_v1["mjd"]) == 0 and len(lc_v2["mjd"]) == 0:
            print(f"  Curvas vacías para physics={physics}")
            return

        # Few-epoch: activar solo si la simulación completa tiene pocas epochs
        n_ep_full1 = _count_epochs(lc_full1, self.epoch_days)
        n_ep_full2 = _count_epochs(lc_full2, self.epoch_days)
        n_ep_v1 = _count_epochs(lc_v1, self.epoch_days)
        n_ep_v2 = _count_epochs(lc_v2, self.epoch_days)
        few_epoch = n_ep_full1 <= _FEW_EPOCH_THRESHOLD or n_ep_full2 <= _FEW_EPOCH_THRESHOLD
        if few_epoch:
            base = lc_full1 if len(lc_full1["mjd"]) >= len(lc_full2["mjd"]) else lc_full2
            lc_v1 = _perturb_within_error(base, rng)
            lc_v2 = _perturb_within_error(base, rng)

        z_val = float(self.z_bins[physics["i_z"]])

        n_ep_full1 = _count_epochs(lc_full1, self.epoch_days)
        n_ep_full2 = _count_epochs(lc_full2, self.epoch_days)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        if few_epoch:
            # Few-epoch: panel izq muestra la base usada, paneles 2-3 las perturbaciones
            panels = [
                (axes[0], base,
                 f"base ({len(base['mjd'])} obs, {_count_epochs(base, self.epoch_days)} ep)", "o"),
                (axes[1], lc_v1, f"vista 1 [perturb] ({len(lc_v1['mjd'])} obs)", "s"),
                (axes[2], lc_v2, f"vista 2 [perturb] ({len(lc_v2['mjd'])} obs)", "^"),
            ]
        else:
            # Normal: cada vista viene de una simulación con distinto t_explosion
            panels = [
                (axes[0], lc_full1,
                 f"sim 1 t_exp={t_exp_1:.0f} ({len(lc_full1['mjd'])} obs, {n_ep_full1} ep)", "o"),
                (axes[1], lc_v1,
                 f"vista 1  {n_visits} visits ({len(lc_v1['mjd'])} obs, {n_ep_v1} ep)", "s"),
                (axes[2], lc_v2,
                 f"vista 2  {n_visits} visits, t_exp={t_exp_2:.0f} ({len(lc_v2['mjd'])} obs, {n_ep_v2} ep)", "^"),
            ]
        for ax, lc, title, marker in panels:
            for local_bi, b in enumerate(self.bands):
                mask = lc["band_idx"] == local_bi
                if not mask.any():
                    continue
                ax.errorbar(
                    lc["mjd"][mask], lc["mag"][mask], yerr=lc["err"][mask],
                    fmt=marker, color=_band_color(b),
                    label=f"{b} ({BAND_LAMBDA_NM.get(b, '?')} nm)",
                    capsize=3, elinewidth=1, alpha=0.85,
                )
            ax.set_title(title)
            ax.set_xlabel("MJD")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("m_AB")
        axes[0].invert_yaxis()

        plt.suptitle(
            f"KN  z={z_val:.4f}  ang={physics['i_angle']}  "
            f"n_visits={n_visits}  {survey_mode}\n"
            f"{self.spec_names[physics['i_spec']]}",
            y=1.02, fontsize=9,
        )
        plt.tight_layout()
        plt.show()
