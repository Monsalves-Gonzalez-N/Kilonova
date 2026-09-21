#!/usr/bin/env bash
# Download OpenUniverse2024 Roman transient snana catalogs (33 healpix, full release).
# Public HTTPS mirror of s3://nasa-irsa-simulations/openuniverse2024/...  (no AWS creds needed).
#
# Two kinds of file per healpix:
#   snana_<healpix>.parquet  -> catalog header (id, gentype, z_CMB, peak_mjd, peak_mag_*). ~4 MB each, ~140 MB total.
#   snana_<healpix>.hdf5     -> per-object `mjd` + true AB mags `mag_{R,Z,Y,J,H,F,K}`.     ~0.5 GB each, ~16 GB total.
#
# THE HDF5 CARRY PHOTOMETRY, NOT SPECTRA. Measured on the 33 files on disk (2026-09-18): every
# object group holds `mjd` and the seven `mag_*`, and nothing else. So there is no `flambda` here to
# de-redshift: the SED templates come from the official model release, not from these files. See
# `scripts/build_openuniverse_cc_templates.py`.
#
# The time axis is NOT the same for every object: measured per gentype, SN Ib/Ic step 1 d, SN II
# and SN Ia step 2 d (up to 4), the Kasen kilonova 0.1 d. `nearest_model_magnitude` snaps to it, so
# the quantisation of `mag_true` is half a step -- up to 1 observer-frame day for a SN II.
#
# Usage:
#   ./download_openuniverse_snana.sh            # all: both .parquet and .hdf5 for all 33 healpix
#   ./download_openuniverse_snana.sh parquet    # light: only .parquet
#   ./download_openuniverse_snana.sh hdf5       # heavy: only .hdf5 (~16 GB)
#   ./download_openuniverse_snana.sh all 9921 9922   # both kinds, only these healpix
#
# TARGET_DIR is where the copy in use lives; override it for a different disk.

set -euo pipefail

TARGET_DIR="${TARGET_DIR:-$HOME/Dropbox/Kilonova/openuniverse2025}"

BASE_URL="https://nasa-irsa-simulations.s3.amazonaws.com/openuniverse2024/roman/full/roman_rubin_cats_v1.1.2_faint"

ALL_HEALPIX=(9921 9922 9923 9924 9925 10050 10051 10052 10053 10177 10178 10179 10180 10181 10305 10306 10307 10308 10429 10430 10431 10432 10549 10550 10551 10552 10665 10666 10667 10668 10777 10778 10779)

KIND="${1:-all}"
case "$KIND" in
  all)     KINDS=(parquet hdf5) ;;
  parquet) KINDS=(parquet) ;;
  hdf5)    KINDS=(hdf5) ;;
  *) echo "First argument must be 'all', 'parquet', or 'hdf5' (got '$KIND')." >&2; exit 1 ;;
esac

shift || true
if [[ $# -gt 0 ]]; then
  HEALPIX=("$@")
else
  HEALPIX=("${ALL_HEALPIX[@]}")
fi

mkdir -p "$TARGET_DIR"

echo "Mode: $KIND  |  kinds: ${KINDS[*]}  |  healpix: ${#HEALPIX[@]}  |  dest: $TARGET_DIR"

# Download all the light parquet first, then the heavy hdf5, so the catalogs are
# in place even if the long hdf5 stage is interrupted.
for kind in "${KINDS[@]}"; do
  for healpix in "${HEALPIX[@]}"; do
    filename="snana_${healpix}.${kind}"
    url="${BASE_URL}/${filename}"
    dest="${TARGET_DIR}/${filename}"
    echo "==> ${filename}"
    # -C - resumes a partial download; --fail aborts on HTTP errors.
    curl --fail --location --continue-at - --output "$dest" "$url"
  done
done

echo "Done. Files in $TARGET_DIR"
