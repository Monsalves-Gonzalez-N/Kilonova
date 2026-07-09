# Roman HLTDS — Filtros y cadencia (Wide vs Deep)

**Fuente:** ROTAC Final Report and Recommendations (Zasowski & Jha et al., April 24, 2025)
arXiv:2505.10574v2 — Sección 3.2 "High Latitude Time Domain Survey (HLTDS)", páginas 10–11.
Archivo local: `main_roman_info.pdf`

## Número de filtros

- **Wide Imaging Tier → 5 filtros: `RZYJH`**
- **Deep Imaging Tier → 5 filtros: `ZYJHF`**

## Cadencia

Ambos tiers alternan dos subconjuntos de filtros: **cada combinación se repite cada ~10 días**,
pero como se intercalan, **una secuencia cae cada ~5 días**.

| Tier | Filtros (5) | Subsecuencias intercaladas | Misma combinación | Secuencia |
|------|-------------|----------------------------|-------------------|-----------|
| Wide | R Z Y J H   | RZY / RJH                  | ~10 días          | ~5 días   |
| Deep | Z Y J H F   | ZYJ / ZHF                  | ~10 días          | ~5 días   |

## Citas textuales (pág. 10–11)

> "The Wide Imaging Tier will cover 10.68 deg² in the North and 7.59 deg² in the
> South with **RZYJH filters**, with a **~10-day cadence of alternating filters**
> (i.e., **one sequence of RZY or RJH every ~5 days**), to reach an average
> maximum-light S/N of 20 for SN Ia at z ~ 0.9."

> "The Deep Imaging Tier will cover 1.97 deg² in the North and 4.5 deg² in the
> South with **ZYJHF filters**, with a similarly **interlaced sequence of ZYJ and
> ZHF images**, to reach an average maximum-light S/N of 20 for SN Ia at z ~ 1.7."

## Notas adicionales

- **Espectroscopía** (Wide 4.5 deg², Deep 0.56 deg²): cadencia **~5 días**;
  exposiciones de 900 s (Wide) y 3600 s (Deep). Ambas en el área deep del Sur.
- SNe Ia "good" (S/N > 40): ~7500 (Wide) y ~6800 (Deep).
- El diseño *overguide* (no recomendado) añadiría imágenes en el Deep Tier en los
  filtros **K y R** para completar el set de filtros completo.
- Pilot Component: 8 observaciones, cadencia 20 días.
- Extended Component: 8 observaciones, cadencia ~120 días.
