"""
Dataset y utilidades para contrastive learning sobre curvas de luz NIR.

Uso típico:
    from funciones import ContrastiveLCDataset, collate_contrastive, worker_init_fn
    from torch.utils.data import DataLoader

    ds = ContrastiveLCDataset(df, objects_raw)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=collate_contrastive,
                        num_workers=4, worker_init_fn=worker_init_fn)

    # Visualizar un ejemplo
    ds.plot(cid=12345)          # objeto concreto
    ds.plot()                   # objeto aleatorio
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Registro de bandas
# ---------------------------------------------------------------------------

# λ_eff en nm. Añadir nuevas bandas aquí; el encoding es automático.
BAND_LAMBDA_NM: dict[str, float] = {
    "J":  1250,
    "H":  1650,
    "Ks": 2150,
    # bandas del dataset Hourglass
    "R":  620,
    "Z":  900,
    "Y":  1020,
    "F":  1850,
}

_LOG_MIN = np.log(300)
_LOG_MAX = np.log(4000)


def _band_enc(band: str) -> float:
    """log(λ) normalizado a [0, 1] sobre [300 nm, 4000 nm]."""
    lam = BAND_LAMBDA_NM.get(band, 1250)
    return (np.log(lam) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)


_BAND_COLOR: dict[str, str] = {
    # Paleta Roman Space Telescope, asignada por λ_eff creciente
    "R":  "#5B8FCC",   # 620 nm  → azul    (F062)
    "Z":  "#44AA55",   # 900 nm  → verde   (F087)
    "Y":  "#CCBB00",   # 1020 nm → amarillo (F106)
    "J":  "#DD8800",   # 1250 nm → naranja  (F129)
    "H":  "#CC3300",   # 1650 nm → rojo-naranja (F158)
    "F":  "#8844AA",   # 1850 nm → violeta  (F184)
    "Ks": "#AA0022",   # 2150 nm → rojo oscuro (F213)
}


def _band_color(band: str) -> str:
    """Color Roman ST por λ_eff. Bandas desconocidas → gris."""
    return _BAND_COLOR.get(band, "#888888")


# ---------------------------------------------------------------------------
# Augmentación contrastiva — versión DataFrame (usada por plot())
# ---------------------------------------------------------------------------

# Pesos sesgados hacia drop alto: favorece vistas parciales (few-shot friendly).
# Objetivo: aprender a clasificar con 1-2 noches de observación.
_PCT_DROP_VALUES  = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
_PCT_DROP_WEIGHTS = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
_PCT_DROP_WEIGHTS /= _PCT_DROP_WEIGHTS.sum()


def _augment_contrastive(lc: pd.DataFrame, rng: np.random.Generator):
    """
    Genera una vista aumentada para contrastive learning (opera sobre DataFrame).
    Usada por ContrastiveLCDataset.plot(). El hot path de entrenamiento usa
    _augment_contrastive_np.

    1. Truncar: elimina un porcentaje aleatorio de observaciones (distribución
       sesgada hacia drop alto → few-shot friendly), ordenadas por MJD.
       Mínimo garantizado: 1 observación.
    2. Perturbar magnitudes: ruido ~ N(0, mag_err).
       Upper limits (mag_err = 0) no se modifican.

    Returns
    -------
    lc_aug : DataFrame
    pct_drop : int
    start : int
    """
    lc = lc.sort_values("mjd").copy()
    n_total = len(lc)

    pct_drop = int(rng.choice(_PCT_DROP_VALUES, p=_PCT_DROP_WEIGHTS))
    n_keep = max(1, int(round(n_total * (1 - pct_drop / 100))))

    max_start = n_total - n_keep
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    lc = lc.iloc[start : start + n_keep].copy()

    noise = rng.normal(loc=0.0, scale=lc["mag_err"].values)
    lc["mag"] = lc["mag"] + noise

    return lc, pct_drop, start


# ---------------------------------------------------------------------------
# Hot path numpy — augmentación + binning vectorizado
# ---------------------------------------------------------------------------

def _df_to_lc_np(lc_df: pd.DataFrame, bands: list[str]) -> dict:
    """
    Convierte un DataFrame de curva de luz (ya ordenado por mjd) a un dict
    de arrays numpy, filtrado a las bandas del proyecto.

    Llamar una vez por objeto en __init__ para eliminar pandas del hot path.
    """
    band_to_idx = {b: i for i, b in enumerate(bands)}
    has_ul = "upper_limit_flag" in lc_df.columns

    mjd      = lc_df["mjd"].values.astype(np.float64)
    mag      = lc_df["mag"].values.astype(np.float32)
    err      = lc_df["mag_err"].values.astype(np.float32)
    ul       = lc_df["upper_limit_flag"].values.astype(np.float32) if has_ul else np.zeros(len(lc_df), np.float32)
    band_idx = np.array([band_to_idx.get(b, -1) for b in lc_df["band"]], dtype=np.int64)

    valid = band_idx >= 0
    mjd_v = mjd[valid]
    # t0 por objeto: origen temporal compartido entre ambas vistas contrastivas,
    # cacheado una vez. Preserva la info de fase relativa entre vistas.
    t0 = float(mjd_v.min()) if len(mjd_v) > 0 else 0.0
    return {
        "mjd":      mjd_v,
        "band_idx": band_idx[valid],
        "mag":      mag[valid],
        "err":      err[valid],
        "ul":       ul[valid],
        "t0":       t0,
    }


def _augment_contrastive_np(lc_np: dict, rng: np.random.Generator):
    """
    Augmentación contrastiva sobre arrays numpy (hot path).

    Los arrays ya están ordenados por mjd (garantizado por df_by_cid en __init__).

    Returns
    -------
    lc_aug : dict de arrays  (vistas del mismo almacenamiento → copy solo en mag)
    pct_drop : int
    start : int
    """
    n_total = len(lc_np["mjd"])

    pct_drop = int(rng.choice(_PCT_DROP_VALUES, p=_PCT_DROP_WEIGHTS))
    n_keep   = max(1, int(round(n_total * (1 - pct_drop / 100))))
    max_start = n_total - n_keep
    start    = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    sl       = slice(start, start + n_keep)

    err = lc_np["err"][sl]
    mag = lc_np["mag"][sl].copy()
    mag += rng.normal(loc=0.0, scale=err).astype(np.float32)

    return {
        "mjd":      lc_np["mjd"][sl],
        "band_idx": lc_np["band_idx"][sl],
        "mag":      mag,
        "err":      err,
        "ul":       lc_np["ul"][sl],
        "t0":       lc_np["t0"],   # heredar t0 del objeto (no recomputar)
    }, pct_drop, start


_T_SCALE    = 35.0    # días de referencia para normalizar t_rel (≈ ventana máxima del survey)
_EPOCH_DAYS = 1.0     # ventana temporal para agrupar observaciones en una época

# Umbral de epochs para activar modo "few-epoch": si una vista tiene
# ≤ este número de epochs, las dos vistas comparten las mismas observaciones
# y solo difieren por perturbación gaussiana dentro del error.
_FEW_EPOCH_THRESHOLD = 2


def _perturb_within_error(lc_np: dict, rng: np.random.Generator) -> dict:
    """
    Genera una vista perturbando magnitudes dentro de la barra de error.
    Usada cuando hay muy pocas epochs para que el windowing genere
    vistas diversas.

    Perturbación: uniforme en [-err, +err] (1σ).  Esto garantiza que
    la magnitud perturbada queda siempre dentro de la barra de error
    mostrada en el plot.
    """
    mag = lc_np["mag"].copy()
    err = lc_np["err"]
    mag += rng.uniform(-err, err).astype(np.float32)
    return {
        "mjd":      lc_np["mjd"],
        "band_idx": lc_np["band_idx"],
        "mag":      mag,
        "err":      lc_np["err"],
        "ul":       lc_np["ul"],
        "t0":       lc_np["t0"],
    }


def _count_epochs(lc_np: dict, epoch_days: float = _EPOCH_DAYS) -> int:
    """Cuenta el número de epochs distintas en un lc_np."""
    if len(lc_np["mjd"]) == 0:
        return 0
    bins = ((lc_np["mjd"] - lc_np["t0"]) / epoch_days).astype(np.int64)
    return len(np.unique(bins))

# Orden canónico de bandas. Determina la estructura del token.
PROJECT_BANDS: list[str] = ["J", "H", "Ks"]

# Bandas por survey Roman
DEEP_BANDS: list[str] = ["Y", "J", "H", "F"]
WIDE_BANDS: list[str] = ["R", "Z", "Y", "J"]
ALL_BANDS:  list[str] = ["R", "Z", "Y", "J", "H", "F"]


def _lc_np_to_epoch_tensor(
    lc_np: dict,
    n_bands: int,
    epoch_days: float = _EPOCH_DAYS,
) -> torch.Tensor:
    """
    Convierte un dict de arrays numpy a tensor de épocas: (n_epochs, token_dim).
    Versión vectorizada sin pandas ni bucles Python.

    Estructura del token
    --------------------
    [ t_rel_norm | mag_J  mag_err_J  ul_J  obs_J
                 | mag_H  mag_err_H  ul_H  obs_H
                 | mag_Ks mag_err_Ks ul_Ks obs_Ks ]

    - t_rel_norm : media de (MJD - t0) / T_SCALE por bin
    - obs_flag   : 1 = banda observada en esta época, 0 = ausente
    - ul_flag    : 1 = upper limit (solo válido si obs_flag=1)
    - Banda no observada → mag=0, mag_err=0, ul=0, obs=0

    token_dim = 1 + 4 * n_bands
    """
    mjd      = lc_np["mjd"]
    band_idx = lc_np["band_idx"]
    mag      = lc_np["mag"]
    err      = lc_np["err"]
    ul       = lc_np["ul"]
    t0       = lc_np["t0"]   # t0 por objeto (compartido entre vistas)

    bins = ((mjd - t0) / epoch_days).astype(np.int64)

    unique_bins = np.unique(bins)
    n_ep        = len(unique_bins)
    bin_to_ep   = np.searchsorted(unique_bins, bins)   # índice de época por obs (0..n_ep-1)

    tok = np.zeros((n_ep, 1 + 4 * n_bands), dtype=np.float32)

    # t_rel_norm: media de (mjd - t0) / T_SCALE por época
    t_rel = ((mjd - t0) / _T_SCALE).astype(np.float32)
    tok[:, 0] = np.bincount(bin_to_ep, weights=t_rel, minlength=n_ep).astype(np.float32)
    tok[:, 0] /= np.bincount(bin_to_ep, minlength=n_ep)

    # Primera observación por (época, banda) — datos ordenados por mjd
    keys       = bin_to_ep * n_bands + band_idx
    _, first   = np.unique(keys, return_index=True)   # primer índice por clave única
    ep_idx     = bin_to_ep[first]
    bi         = band_idx[first]

    tok[ep_idx, 1 + 4 * bi + 0] = mag[first]
    tok[ep_idx, 1 + 4 * bi + 1] = err[first]
    tok[ep_idx, 1 + 4 * bi + 2] = ul[first]
    tok[ep_idx, 1 + 4 * bi + 3] = 1.0

    return torch.from_numpy(tok)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveLCDataset(Dataset):
    """
    Dataset PyTorch para contrastive learning sobre curvas de luz.

    Cada item devuelve un par de vistas del mismo objeto:
        x1    : (n_epochs,  token_dim)  — ventana temporal con offset t0_1
        x2    : (n_epochs', token_dim)  — ventana temporal con offset t0_2
        z     : float                   — redshift z_cmb (global, no por token)
        label : int

    Vistas contrastivas: misma SN, distinto "timing de suerte" (t0 offset
    dentro de [0, CADENCE) días). Sin masking artificial de bandas — la
    non-detection es señal astrofísica. Ventana de 2-7 visitas (~10-35 días)
    para ser consistente con la duración observable de kilonovas.

    Token = 1 época = [t_rel_norm | mag_b1  err_b1  ul_b1  obs_b1 | ...]
    token_dim = 1 + 4 * len(bands)

    Parameters
    ----------
    df : DataFrame
        Fotometría. Columnas: mjd, band, mag, mag_err, cid.
    objects_raw : DataFrame
        Indexado por cid. Columnas: class, z_cmb, peak_mjd.
    bands : list[str], default PROJECT_BANDS
        Bandas a incluir en cada token (orden canónico).
    epoch_days : float, default 1.0
        Ventana temporal en días para agrupar observaciones en una época.
    z_dropout : float, default 0.2
        Prob. de enmascarar z=0 durante training. 0.0 en inferencia.
    seed : int, optional
    """

    # Cadencia fija de 5 días — mismos MJDs que Roman
    CADENCE = 5.0
    # Rango de visitas por ventana (2-7, consistente con KN observables)
    N_VISITS_RANGE = (2, 7)
    # Mismo corte que en KN dataset: obs con mag_err >= 2 se descartan
    MAX_MAG_ERR = 2.0

    def __init__(
        self,
        df: pd.DataFrame,
        objects_raw: pd.DataFrame,
        bands: list[str] = PROJECT_BANDS,
        epoch_days: float = _EPOCH_DAYS,
        z_dropout: float = 0.2,
        label: int | None = None,
        seed: int | None = None,
    ):
        self.bands      = bands
        self.epoch_days = epoch_days
        self.z_dropout  = z_dropout
        self._fixed_label = label  # Si no es None, todos los objetos usan este label

        classes = sorted(objects_raw["class"].dropna().unique())
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.objects_raw = objects_raw
        self._rng = np.random.default_rng(seed)

        # Pre-computar lc_np por objeto. El df debe llegar ya filtrado
        # (fluxcal>0, z_max, bandas cubiertas, etc.) desde el notebook.
        self._lc_np: dict = {}
        for cid, g in df.groupby("cid", sort=False):
            if cid not in objects_raw.index:
                continue
            g_sorted = g.sort_values("mjd").reset_index(drop=True)
            self._lc_np[cid] = _df_to_lc_np(g_sorted, bands)

        self.cids = np.array(list(self._lc_np.keys()))
        print(f"ContrastiveLCDataset: {len(self.cids)} objetos")

    def __len__(self) -> int:
        return len(self.cids)

    def _window_view(self, lc_np: dict, mjd_start: float, n_visits: int) -> dict:
        """
        Extrae una ventana temporal de n_visits épocas (cada CADENCE días).

        Parameters
        ----------
        lc_np : dict — curva completa en formato numpy
        mjd_start : float — MJD de inicio de la ventana
        n_visits : int — número de visitas (2-7)
        """
        mjd_end = mjd_start + n_visits * self.CADENCE
        mask = ((lc_np["mjd"] >= mjd_start) & (lc_np["mjd"] < mjd_end)
                & (lc_np["err"] < self.MAX_MAG_ERR))

        return {
            "mjd":      lc_np["mjd"][mask],
            "band_idx": lc_np["band_idx"][mask],
            "mag":      lc_np["mag"][mask],
            "err":      lc_np["err"][mask],
            "ul":       lc_np["ul"][mask],
            "t0":       float(lc_np["mjd"][mask][0]) if mask.any() else lc_np["t0"],
        }

    def __getitem__(self, idx: int):
        cid = self.cids[idx]
        lc_np = self._lc_np[cid]

        # RNG per-item: reproducible y thread-safe con num_workers > 0
        rng = np.random.default_rng(self._rng.integers(0, 2**63 - 1))

        n_visits = int(rng.integers(self.N_VISITS_RANGE[0],
                                    self.N_VISITS_RANGE[1] + 1))
        window_days = n_visits * self.CADENCE

        # Rango válido para el inicio de la ventana: cualquier punto donde
        # quepan n_visits épocas dentro de las observaciones existentes
        mjd_min = lc_np["mjd"][0]
        mjd_max = lc_np["mjd"][-1]
        latest_start = mjd_max - window_days

        if latest_start < mjd_min:
            latest_start = mjd_min

        # Dos vistas: misma SN, distinto punto de inicio aleatorio
        start_1 = float(rng.uniform(mjd_min, latest_start + self.CADENCE))
        start_2 = float(rng.uniform(mjd_min, latest_start + self.CADENCE))

        lc1 = self._window_view(lc_np, start_1, n_visits)
        lc2 = self._window_view(lc_np, start_2, n_visits)

        # Fallback: si alguna vista quedó vacía, usar toda la ventana máxima
        if len(lc1["mjd"]) < 1 or len(lc2["mjd"]) < 1:
            lc1 = self._window_view(lc_np, mjd_min, self.N_VISITS_RANGE[1])
            lc2 = self._window_view(lc_np, mjd_min, self.N_VISITS_RANGE[1])

        # Few-epoch: activar solo si la curva completa del objeto tiene pocas
        # epochs (evento intrínsecamente corto). No activar si la curva es
        # larga y solo la view cayó en pocas epochs por mala suerte del window.
        n_ep_full = _count_epochs(lc_np, self.epoch_days)
        if n_ep_full <= _FEW_EPOCH_THRESHOLD:
            lc1 = _perturb_within_error(lc_np, rng)
            lc2 = _perturb_within_error(lc_np, rng)

        x1 = _lc_np_to_epoch_tensor(lc1, len(self.bands), self.epoch_days)
        x2 = _lc_np_to_epoch_tensor(lc2, len(self.bands), self.epoch_days)

        # Guard: tensor vacío → token mínimo (1 época, ceros)
        token_dim = 1 + 4 * len(self.bands)
        if x1.shape[0] == 0:
            x1 = torch.zeros(1, token_dim)
        if x2.shape[0] == 0:
            x2 = torch.zeros(1, token_dim)

        obj_row = self.objects_raw.loc[cid]
        z_true  = float(obj_row["z_cmb"]) if "z_cmb" in obj_row.index else 0.0
        label   = self._fixed_label if self._fixed_label is not None else self.class_to_idx.get(obj_row["class"], -1)

        z = 0.0 if (self.z_dropout > 0 and rng.random() < self.z_dropout) else z_true
        return x1, x2, float(z), label

    # ── Visualización ─────────────────────────────────────────────────────

    def plot(self, cid=None, seed: int | None = None):
        """
        Muestra dos vistas contrastivas del mismo objeto (ventanas temporales
        con distinto punto de inicio aleatorio).
        """
        rng = np.random.default_rng(seed)

        if cid is None:
            cid = rng.choice(self.cids)

        lc_np = self._lc_np[cid]
        n_visits = int(rng.integers(self.N_VISITS_RANGE[0],
                                    self.N_VISITS_RANGE[1] + 1))
        window_days = n_visits * self.CADENCE

        mjd_min = lc_np["mjd"][0]
        mjd_max = lc_np["mjd"][-1]
        latest_start = max(mjd_min, mjd_max - window_days)

        start_1 = float(rng.uniform(mjd_min, latest_start + self.CADENCE))
        start_2 = float(rng.uniform(mjd_min, latest_start + self.CADENCE))
        lc_v1 = self._window_view(lc_np, start_1, n_visits)
        lc_v2 = self._window_view(lc_np, start_2, n_visits)

        # Few-epoch: activar solo si la curva completa tiene pocas epochs
        n_ep_full = _count_epochs(lc_np, self.epoch_days)
        few_epoch = n_ep_full <= _FEW_EPOCH_THRESHOLD
        if few_epoch:
            lc_v1 = _perturb_within_error(lc_np, rng)
            lc_v2 = _perturb_within_error(lc_np, rng)

        obj_row = self.objects_raw.loc[cid] if cid in self.objects_raw.index else None
        object_class = obj_row["class"] if obj_row is not None else "?"

        MAX_ERR_PLOT = 2.0  # obs con err >= esto se muestran como non-detection

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        few_tag = " [few-epoch]" if few_epoch else ""
        panels = [
            (axes[0], lc_np, f"completa ({len(lc_np['mjd'])} obs)", "o"),
            (axes[1], lc_v1, f"vista 1{few_tag} ({len(lc_v1['mjd'])} obs)", "s"),
            (axes[2], lc_v2, f"vista 2{few_tag} ({len(lc_v2['mjd'])} obs)", "^"),
        ]
        for ax, lc, title, marker in panels:
            for bi, band in enumerate(self.bands):
                bmask = lc["band_idx"] == bi
                if not bmask.any():
                    continue
                color = _band_color(band)
                lbl = f"{band} ({BAND_LAMBDA_NM.get(band, '?')} nm)"

                # Detecciones: err < MAX_ERR_PLOT
                det = bmask & (lc["err"] < MAX_ERR_PLOT)
                if det.any():
                    ax.errorbar(
                        lc["mjd"][det], lc["mag"][det], yerr=lc["err"][det],
                        fmt=marker, color=color,
                        label=lbl,
                        capsize=3, elinewidth=1, alpha=0.85,
                    )
                # Non-detections: err >= MAX_ERR_PLOT
                ndet = bmask & (lc["err"] >= MAX_ERR_PLOT)
                if ndet.any():
                    ax.scatter(
                        lc["mjd"][ndet], lc["mag"][ndet],
                        marker="v", color=color, alpha=0.4, s=30,
                        label=f"{band} non-det" if not det.any() else None,
                    )
            ax.set_title(f"CID {cid} — {title}")
            ax.set_xlabel("MJD")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("mag")
        axes[0].invert_yaxis()

        plt.suptitle(
            f"{object_class}  |  n_visits={n_visits}  |  cid={cid}",
            y=1.02, fontsize=12,
        )
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Worker init — reseed RNG por worker para diversidad de augmentación
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int):
    """
    Pasar como worker_init_fn al DataLoader cuando num_workers > 0.

    Sin esto, todos los workers heredan el mismo estado de RNG (fork) y generan
    las mismas augmentaciones, reduciendo la diversidad contrastiva.

    Soporta ConcatDataset: reseedea _rng en cada dataset interno.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = info.seed % (2**32)
    dataset = info.dataset
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for i, ds in enumerate(dataset.datasets):
            if hasattr(ds, '_rng'):
                ds._rng = np.random.default_rng(seed + i)
    elif hasattr(dataset, '_rng'):
        dataset._rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Collate function para DataLoader
# ---------------------------------------------------------------------------

def collate_contrastive(batch):
    """
    Agrupa un batch de (x1, x2, z, label) con padding a la longitud máxima.

    x1 y x2 se padean a la misma longitud L = max(L1_max, L2_max) para
    permitir el forward combinado en LightCurveEncoder.

    Returns
    -------
    x1_pad   : (B, L, token_dim)   — token_dim = 1 + 4 * n_bands
    x2_pad   : (B, L, token_dim)
    mask1    : (B, L)  — True donde hay padding
    mask2    : (B, L)  — True donde hay padding
    redshifts: (B,)
    labels   : (B,)
    """
    x1s, x2s, zs, labels = zip(*batch)

    L = max(
        max(s.shape[0] for s in x1s),
        max(s.shape[0] for s in x2s),
    )

    def pad(seqs):
        padded = torch.zeros(len(seqs), L, seqs[0].shape[1])
        mask   = torch.ones(len(seqs), L, dtype=torch.bool)
        for i, s in enumerate(seqs):
            n = s.shape[0]
            padded[i, :n] = s
            mask[i, :n]   = False   # False = posición válida
        return padded, mask

    x1_pad, mask1 = pad(x1s)
    x2_pad, mask2 = pad(x2s)
    redshifts = torch.tensor(zs, dtype=torch.float32)
    labels    = torch.tensor(labels, dtype=torch.long)

    return x1_pad, x2_pad, mask1, mask2, redshifts, labels


# ---------------------------------------------------------------------------
# Encoder contrastivo para curvas de luz NIR
# ---------------------------------------------------------------------------

class DecayTimeEmbed(nn.Module):
    """
    Encoding temporal con bases de decaimiento exponencial.

    d(t) = [1, w₀·t + φ₀, exp(-α₁·t), …, exp(-αₖ·t)]
    Output: (*, k+2)

    Motivación: curvas de luz de transientes son decaimientos, no señales
    periódicas. Las α aprendibles convergen a las escalas de tiempo físicas
    (KN ~días, SN Ia ~semanas, SN II ~meses).

    Inicialización: α ∈ [0.5, 20] en log-space, cubriendo desde decaimientos
    lentos (apenas cambia en la ventana) hasta rápidos (decae en 1-2 épocas).
    Con _T_SCALE=35 días, t_rel ∈ [0, ~1].
    """
    def __init__(self, k: int = 8):
        super().__init__()
        self.w0   = nn.Parameter(torch.randn(1))
        self.phi0 = nn.Parameter(torch.randn(1))
        self.log_alpha = nn.Parameter(
            torch.linspace(math.log(0.5), math.log(20.0), k)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)                          # (*, 1)
        alpha = self.log_alpha.exp()                  # (k,)
        exponent = (-alpha * t).clamp(min=-80.0)      # (*, k)
        return torch.cat([
            torch.ones_like(t),                       # base constante
            t * self.w0 + self.phi0,                  # base lineal
            torch.exp(exponent),                      # bases de decaimiento
        ], dim=-1)                                    # (*, k+2)


class LightCurveEncoder(nn.Module):
    """
    Encoder contrastivo para curvas de luz NIR (Roman).

    Objetivo: aprender representaciones invariantes a la cobertura temporal.
    Curva completa y subregión del mismo objeto → mismo embedding.
    Diseñado para operar desde el inicio del survey con curvas parciales.

    Token por época
    ---------------
    [ t_rel_norm | (mag, mag_err, ul_flag, obs_flag) × n_bands ]
    t_rel_norm → DecayTimeEmbed(k) → (k+2) features, luego Linear → d_model

    Token z (condicionamiento global, opcional)
    ------------------------------------------
    [z_phot | σ_z | flag_host] → Linear → d_model
    Maskeado con z_dropout durante training → robusto a ausencia de host.

    Uso futuro
    ----------
    encode() devuelve el CLS bruto (d_model) reutilizable para clasificación
    supervisada o anomaly detection (MAE) sin reentrenar el encoder.

    Parameters
    ----------
    bands : list[str], default PROJECT_BANDS
        Debe coincidir con el bands usado en ContrastiveLCDataset.
    d_model : int, default 128
    n_heads : int, default 8
    n_layers : int, default 4
    t2v_k : int, default 8
    proj_dim : int, default 64   — espacio de proyección NT-Xent
    z_dropout : float, default 0.3
    """

    def __init__(
        self,
        bands: list[str] = PROJECT_BANDS,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        t2v_k: int = 8,
        proj_dim: int = 64,
        z_dropout: float = 0.3,
    ):
        super().__init__()
        self.z_dropout = z_dropout
        self.n_bands   = len(bands)

        self.t2v = DecayTimeEmbed(k=t2v_k)

        # token_dim = 1 (t_rel) + 4 * n_bands; DecayTimeEmbed expande t_rel → (t2v_k+2)
        in_dim = (t2v_k + 2) + 4 * self.n_bands
        self.token_proj = nn.Linear(in_dim, d_model)
        self.z_proj     = nn.Linear(3, d_model)   # [z_phot, σ_z, flag_host]
        self.cls_token  = nn.Parameter(torch.randn(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1 + 4*n_bands)
        # x[..., 0]  = t_rel_norm  → t2v → (B, L, t2v_k+2)
        # x[..., 1:] = per-band features (mag, err, ul, obs) × n_bands
        raw = torch.cat([self.t2v(x[..., 0]), x[..., 1:]], dim=-1)
        return self.token_proj(raw)   # (B, L, d_model)

    def _z_token(self, z, sigma_z, flag_host) -> torch.Tensor:
        B, dev = z.shape[0], z.device
        sz = sigma_z   if sigma_z   is not None else torch.zeros(B, device=dev)
        fh = flag_host if flag_host is not None else (z > 0).float()
        return self.z_proj(torch.stack([z, sz, fh], dim=-1)).unsqueeze(1)

    def _apply_z_dropout(self, z, sigma_z, flag_host):
        """Aplica z_dropout una sola vez antes de encode para que ambas vistas sean consistentes."""
        if self.training and self.z_dropout > 0:
            drop = torch.rand(z.shape[0], device=z.device) < self.z_dropout
            z = z.clone(); z[drop] = 0.0
            if sigma_z is not None:
                sigma_z = sigma_z.clone(); sigma_z[drop] = 0.0
            if flag_host is not None:
                flag_host = flag_host.clone(); flag_host[drop] = 0.0
        return z, sigma_z, flag_host

    def encode(
        self,
        x: torch.Tensor,         # (B, L, token_dim)
        pad_mask: torch.Tensor,  # (B, L) True=padding
        z: torch.Tensor,         # (B,)
        sigma_z: torch.Tensor | None = None,
        flag_host: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode una curva de luz (completa o parcial) → representación CLS.

        Secuencia: [CLS] | z_tok | obs_tokens → Transformer

        Returns
        -------
        cls : (B, d_model)
        """
        B = x.shape[0]
        cls    = self.cls_token.unsqueeze(0).expand(B, -1).unsqueeze(1)
        z_tok  = self._z_token(z, sigma_z, flag_host)
        tokens = self._tokenize(x)

        enc_in  = torch.cat([cls, z_tok, tokens], dim=1)
        enc_pad = torch.cat([
            torch.zeros(B, 2, dtype=torch.bool, device=x.device), pad_mask
        ], dim=1)

        return self.encoder(enc_in, src_key_padding_mask=enc_pad)[:, 0, :]  # CLS output

    def forward(
        self,
        x1: torch.Tensor, pad_mask1: torch.Tensor,
        x2: torch.Tensor, pad_mask2: torch.Tensor,
        z: torch.Tensor,
        sigma_z: torch.Tensor | None = None,
        flag_host: torch.Tensor | None = None,
    ):
        """
        Forward contrastivo: encodea ambas vistas con un único forward pass.

        x1 y x2 deben tener la misma longitud L (garantizado por collate_contrastive).
        Se concatenan en la dimensión batch → (2B, L, D) → un solo encoder call
        → mejora utilización GPU vs dos forwards separados.

        z_dropout aplicado UNA vez → ambas vistas ven el mismo z (o ausencia).

        Returns
        -------
        h1, h2 : (B, proj_dim) — proyecciones L2-normalizadas para NT-Xent
        """
        z, sigma_z, flag_host = self._apply_z_dropout(z, sigma_z, flag_host)

        # Forward combinado: (2B, L, token_dim)
        x_cat    = torch.cat([x1, x2], dim=0)
        mask_cat = torch.cat([pad_mask1, pad_mask2], dim=0)
        z_cat    = z.repeat(2)
        sz_cat   = sigma_z.repeat(2)    if sigma_z   is not None else None
        fh_cat   = flag_host.repeat(2)  if flag_host is not None else None

        cls      = self.encode(x_cat, mask_cat, z_cat, sz_cat, fh_cat)
        cls1, cls2 = cls.chunk(2, dim=0)

        h1 = nn.functional.normalize(self.proj_head(cls1), dim=-1)
        h2 = nn.functional.normalize(self.proj_head(cls2), dim=-1)
        return h1, h2


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def nt_xent_loss(
    h1: torch.Tensor,          # (B, proj_dim) — normalizado
    h2: torch.Tensor,          # (B, proj_dim) — normalizado
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    NT-Xent loss (SimCLR). Pares positivos: (h1[i], h2[i]).
    Negativos: todos los demás items del batch.

    Con B objetos hay 2B vectores y 2(B-1) negativos por par.

    Nota: con AMP (float16/bfloat16) usar temperature suficientemente grande
    (≥0.05) para evitar overflow en la matriz de similitud.

    Returns
    -------
    loss : escalar
    """
    B = h1.shape[0]
    assert B > 1, f"NT-Xent requires batch size > 1, got B={B}"
    h   = torch.cat([h1, h2], dim=0)
    sim = torch.mm(h, h.t()) / temperature

    mask_self = torch.eye(2 * B, dtype=torch.bool, device=h.device)
    sim.masked_fill_(mask_self, -1e9)

    labels = torch.cat([
        torch.arange(B, 2 * B, device=h.device),
        torch.arange(0, B,     device=h.device),
    ])
    return nn.functional.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# DataLoader combinado KN + SN (binario: KN=0, nonKN=1)
# ---------------------------------------------------------------------------

def build_combined_loader(
    sn_df: pd.DataFrame,
    sn_objects: pd.DataFrame,
    survey_mjds: dict[str, np.ndarray],
    kn_grid_dir: str,
    kn_n_virtual: int = 100_000,
    kn_t_max: float = 60.0,
    min_bands: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    epoch_days: float = _EPOCH_DAYS,
    z_dropout: float = 0.2,
    seed: int | None = None,
) -> tuple:
    """
    Construye un DataLoader combinado KN (label=0) + nonKN (label=1)
    con token de 6 bandas (ALL_BANDS = R, Z, Y, J, H, F).

    Cada objeto (SN o KN) tiene observaciones en las 4 bandas de su survey
    (WIDE: R,Z,Y,J  o  DEEP: Y,J,H,F). Las 2 bandas del otro modo quedan
    con obs_flag=0 en el token — consistente con la restricción del survey
    Roman (una noche es WIDE o DEEP, nunca ambos).

    Las KN eligen modo WIDE o DEEP aleatoriamente por sample.
    Las SN se filtran a objetos con ≥ min_bands bandas en las 6 bandas.

    Parameters
    ----------
    sn_df : DataFrame
        Fotometría SN con columnas: cid, band, mjd, mag, mag_err.
    sn_objects : DataFrame
        Indexado por cid. Columnas: class, z_cmb.
    survey_mjds : dict[str, np.ndarray]
        MJDs únicos del survey por banda (las 6 bandas de ALL_BANDS).
    kn_grid_dir : str
        Path a la grid pre-computada de KN.
    min_bands : int
        Mínimo de bandas distintas por objeto SN (default 4).

    Returns
    -------
    loader : DataLoader
    ds_kn  : KilonovaContrastiveDataset
    ds_sn  : ContrastiveLCDataset
    """
    import sys
    _kn_dir = str(Path(__file__).resolve().parent.parent / "kn")
    sys.path.insert(0, _kn_dir)
    from kn_dataset import KilonovaContrastiveDataset

    bands = ALL_BANDS

    # Filtrar SN: solo bandas válidas y objetos con ≥ min_bands
    sn_df_filt = sn_df[sn_df["band"].isin(bands)].copy()
    bands_per_cid = sn_df_filt.groupby("cid")["band"].nunique()
    valid_cids = bands_per_cid[bands_per_cid >= min_bands].index
    sn_df_filt = sn_df_filt[sn_df_filt["cid"].isin(valid_cids)].reset_index(drop=True)

    ds_kn = KilonovaContrastiveDataset(
        grid_dir=kn_grid_dir,
        survey_mjds=survey_mjds,
        t_max=kn_t_max,
        n_virtual=kn_n_virtual,
        label=0,           # KN = 0
        epoch_days=epoch_days,
        z_dropout=z_dropout,
        seed=seed,
    )

    ds_sn = ContrastiveLCDataset(
        df=sn_df_filt,
        objects_raw=sn_objects,
        bands=bands,
        epoch_days=epoch_days,
        z_dropout=z_dropout,
        label=1,           # nonKN = 1
        seed=seed,
    )

    combined = torch.utils.data.ConcatDataset([ds_kn, ds_sn])

    # Pesos para balancear: cada clase contribuye ~50% del batch
    n_kn  = len(ds_kn)
    n_sn  = len(ds_sn)
    w_kn  = 1.0 / n_kn
    w_sn  = 1.0 / n_sn
    weights = [w_kn] * n_kn + [w_sn] * n_sn
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(combined), replacement=True,
    )

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_contrastive,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    return loader, ds_kn, ds_sn


def build_split_loaders(
    sn_df: pd.DataFrame,
    sn_objects: pd.DataFrame,
    survey_mjds: dict[str, np.ndarray],
    kn_grid_dir: str,
    kn_n_virtual: tuple[int, int, int] = (80_000, 10_000, 10_000),
    kn_t_max: float = 60.0,
    min_bands: int = 4,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    batch_size: int = 32,
    num_workers: int = 4,
    epoch_days: float = _EPOCH_DAYS,
    z_dropout: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Construye loaders de train/val/test con splits sin data leakage.

    Split strategy:
        - SN: por CID único (un objeto solo aparece en un split)
        - KN: por modelo (spec_index). Todos los ángulos de un modelo
          van al mismo split. Evita que el modelo memorice espectros.

    Parameters
    ----------
    kn_n_virtual : (train, val, test)
        Tamaño virtual del dataset KN por split.
    split_ratios : (train, val, test)
        Fracciones para SN CIDs y KN modelos.

    Returns
    -------
    dict con keys "train", "val", "test", cada uno con:
        {"loader": DataLoader, "ds_kn": ..., "ds_sn": ...}
    """
    import sys
    _kn_dir = str(Path(__file__).resolve().parent.parent / "kn")
    sys.path.insert(0, _kn_dir)
    from kn_dataset import KilonovaContrastiveDataset

    rng = np.random.default_rng(seed)
    bands = ALL_BANDS

    # ── Filtrar SN ────────────────────────────────────────────────────────
    sn_df_filt = sn_df[sn_df["band"].isin(bands)].copy()
    bands_per_cid = sn_df_filt.groupby("cid")["band"].nunique()
    valid_cids = bands_per_cid[bands_per_cid >= min_bands].index
    sn_df_filt = sn_df_filt[sn_df_filt["cid"].isin(valid_cids)].reset_index(drop=True)

    # ── Split SN por CID ──────────────────────────────────────────────────
    all_cids = sn_df_filt["cid"].unique()
    rng.shuffle(all_cids)

    n_train = int(len(all_cids) * split_ratios[0])
    n_val   = int(len(all_cids) * split_ratios[1])
    cids_split = {
        "train": set(all_cids[:n_train]),
        "val":   set(all_cids[n_train:n_train + n_val]),
        "test":  set(all_cids[n_train + n_val:]),
    }

    # ── Split KN por modelo (spec_index) ──────────────────────────────────
    mags_tmp = np.load(Path(kn_grid_dir) / "mags.npy", mmap_mode="r")
    n_spec_total = mags_tmp.shape[0]
    spec_all = np.arange(n_spec_total)
    rng.shuffle(spec_all)

    n_train_kn = int(n_spec_total * split_ratios[0])
    n_val_kn   = int(n_spec_total * split_ratios[1])
    spec_split = {
        "train": spec_all[:n_train_kn],
        "val":   spec_all[n_train_kn:n_train_kn + n_val_kn],
        "test":  spec_all[n_train_kn + n_val_kn:],
    }

    print(f"Split SN CIDs:  train={n_train}  val={n_val}  test={len(all_cids)-n_train-n_val}")
    print(f"Split KN specs: train={len(spec_split['train'])}  "
          f"val={len(spec_split['val'])}  test={len(spec_split['test'])}")

    # ── Construir loaders ─────────────────────────────────────────────────
    result = {}
    for i, split in enumerate(["train", "val", "test"]):
        # SN: filtrar por CIDs del split
        mask = sn_df_filt["cid"].isin(cids_split[split])
        df_split = sn_df_filt[mask].reset_index(drop=True)

        # z_dropout solo en train
        zd = z_dropout if split == "train" else 0.0

        is_train = (split == "train")

        ds_kn = KilonovaContrastiveDataset(
            grid_dir=kn_grid_dir,
            survey_mjds=survey_mjds,
            spec_indices=spec_split[split],
            t_max=kn_t_max,
            n_virtual=kn_n_virtual[i] if is_train else None,
            deterministic=not is_train,
            label=0,
            epoch_days=epoch_days,
            z_dropout=zd,
            seed=seed + i,
        )

        ds_sn = ContrastiveLCDataset(
            df=df_split,
            objects_raw=sn_objects,
            bands=bands,
            epoch_days=epoch_days,
            z_dropout=zd,
            label=1,
            seed=seed + i,
        )

        combined = torch.utils.data.ConcatDataset([ds_kn, ds_sn])

        n_kn, n_sn = len(ds_kn), len(ds_sn)
        if is_train:
            # Train: sampler balanceado + shuffle implícito
            w_kn, w_sn = 1.0 / n_kn, 1.0 / n_sn
            weights = [w_kn] * n_kn + [w_sn] * n_sn
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(combined), replacement=True,
            )
            shuffle = False
        else:
            # Val/test: sin replacement, shuffle para mezclar KN y SN en cada batch
            # (ConcatDataset es secuencial: sin shuffle los batches son homogéneos)
            sampler = None
            shuffle = True

        loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_contrastive,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

        result[split] = {"loader": loader, "ds_kn": ds_kn, "ds_sn": ds_sn}

    return result
