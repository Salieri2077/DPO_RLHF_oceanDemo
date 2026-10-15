"""Pinned access to OceanDepths (ESA-philab/OceanDepths) with qc_policy_v1 and partition_v1.

Rules follow Agent Evaluation/SKILLRL_OCEAN_DESIGN.md section 2/5 and OCEANDEPTHS_M0.md section 6:
- the index (profiles.parquet) is the source of glorys_temporal_status: zero-only Zarr chunks of that array are absent
  with fill_value=null, so slicing it returns uninitialised bytes;
- every Zarr chunk that is read must exist (absent chunks would silently become fill values);
- masks are int8 0/1 and are compared with == 1;
- qc_policy_v1: finite time, valid coordinates, position QC = 1, the analysed variable's profile QC = 1, level QC = 1,
  mask == 1 and a finite value; time / profile-depth / level-depth QC must be 1 when available, -1 means unknown.
"""
import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/anhuang/oceandepths")
REVISION = "d8c813ebd6126066ebe8a13b4c6bae769c8e0a07"
STORE = "data/argo_glors_ostia_ssh.zarr"
JULD_EPOCH = pd.Timestamp("1950-01-01")
QC_POLICY = "qc_policy_v1"
PARTITION = "partition_v1"

VARIABLES = {  # analysed quantity -> EN4 value / mask / level QC / profile QC arrays and its GLORYS counterpart
    "potential_temperature": dict(value="argo_potm_on_glorys_depth", mask="argo_potm_valid_on_glorys_depth",
                                  level_qc="argo_potm_qc_on_glorys_depth", profile_qc="argo_profile_potm_qc",
                                  reanalysis="glorys_thetao", units="degC"),
    "salinity": dict(value="argo_psal_on_glorys_depth", mask="argo_psal_valid_on_glorys_depth",
                     level_qc="argo_psal_qc_on_glorys_depth", profile_qc="argo_profile_psal_qc",
                     reanalysis="glorys_so", units="PSU"),
}
# Inspection only: the store has no profile-level QC for in-situ temperature, and GLORYS thetao is potential
# temperature, so neither method card accepts it.
INSPECT_ONLY = {
    "in_situ_temperature": dict(value="argo_temp_on_glorys_depth", mask="argo_temp_valid_on_glorys_depth",
                                level_qc="argo_temp_qc_on_glorys_depth", profile_qc=None, reanalysis=None,
                                units="degC"),
}
QC_POLICIES = (QC_POLICY, "finite_only")  # finite_only (mask == 1 and finite) is a diagnostic, never a method card

# partition_v1 (frozen 2026-10-10): years, 30-day buffers on both sides of each year boundary, 10x10 degree blocks and
# a reserved contiguous region for spatial generalisation (South Atlantic) with a 1-degree buffer.
SPLIT_YEARS = {"train": (2000, 2016), "dev": (2017, 2019), "test": (2020, 2024)}
SPLIT_BOUNDARIES = [pd.Timestamp("2017-01-01"), pd.Timestamp("2020-01-01")]
TIME_BUFFER_DAYS = 30
RESERVED_REGION = {"name": "South Atlantic 40S-20S, 40W-0", "lat": (-40.0, -20.0), "lon": (-40.0, 0.0)}
SPATIAL_BUFFER_DEG = 1.0
BLOCK_DEG = 10.0


@dataclass(frozen=True)
class Profiles:
    """Rows of the store (sorted) plus the per-profile index columns they came with."""
    rows: np.ndarray
    index: pd.DataFrame


class OceanDepths:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        meta = json.loads((self.root / "SOURCE_MANIFEST.json").read_text())
        if meta["revision"] != REVISION:
            raise ValueError(f"expected revision {REVISION}, found {meta['revision']}")

    @cached_property
    def store(self):
        import zarr
        return zarr.open_consolidated(str(self.root / STORE), mode="r")

    @cached_property
    def index(self):
        frame = pd.read_parquet(self.root / "indices/profiles.parquet")
        if not (frame["profile"].to_numpy() == np.arange(len(frame))).all():
            raise ValueError("profiles.parquet is not in store row order")
        if not frame["glorys_temporal_status"].isin([0, 1, 2]).all():
            raise ValueError("unexpected glorys_temporal_status code in the index")
        frame["time"] = JULD_EPOCH + pd.to_timedelta(frame["profile_juld"], unit="D")
        return frame

    @cached_property
    def depth(self):
        depth = self.store["glorys_depth"][:].astype(np.float64)
        if not (np.all(np.isfinite(depth)) and np.all(np.diff(depth) > 0)):
            raise ValueError("depth axis must be finite and strictly increasing")
        return depth

    def read(self, name, rows):
        """Rows (sorted ints) of one array; refuses to read through an absent chunk."""
        array = self.store[name]
        if len(rows) == 0:
            return np.empty((0, *array.shape[1:]), dtype=array.dtype)
        chunk = array.chunks[0]
        suffix = ".0" if array.ndim == 2 else ""
        folder = self.root / STORE / name
        missing = [c for c in np.unique(rows // chunk) if not (folder / f"{c}{suffix}").is_file()]
        if missing:
            raise ValueError(f"{name}: chunks {missing[:5]} are absent; their values would be fill bytes")
        if array.ndim == 2:
            return array.get_orthogonal_selection((rows, slice(None)))
        return array.get_orthogonal_selection(rows)

    def select(self, lat, lon, start, end):
        """Profiles inside [lat0, lat1) x [lon0, lon1) observed on dates start..end (inclusive, YYYY-MM-DD)."""
        frame = self.index
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        mask = (frame["latitude"].between(lat[0], lat[1], inclusive="left")
                & frame["longitude"].between(lon[0], lon[1], inclusive="left")
                & (frame["time"] >= t0) & (frame["time"] < t1))
        part = frame[mask]
        return Profiles(part["profile"].to_numpy(), part)

    def profile_ok(self, profiles, variable):
        """qc_policy_v1 at profile level -> (bool array, reasons per profile)."""
        spec, rows, idx = VARIABLES[variable], profiles.rows, profiles.index
        position = self.read("argo_position_qc", rows)
        variable_qc = self.read(spec["profile_qc"], rows)
        juld_qc = self.read("argo_juld_qc", rows)
        depth_qc = self.read("argo_profile_depth_qc", rows)
        lat, lon = idx["latitude"].to_numpy(), idx["longitude"].to_numpy()
        coords = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (lon >= -180) & (lon < 180)
        time = np.isfinite(idx["profile_juld"].to_numpy())
        optional = ((juld_qc == 1) | (juld_qc == -1)) & ((depth_qc == 1) | (depth_qc == -1))
        ok = coords & time & (position == 1) & (variable_qc == 1) & optional
        return ok, {"position_qc": position, "profile_qc": variable_qc, "juld_qc": juld_qc,
                    "profile_depth_qc": depth_qc, "coords_ok": coords, "time_ok": time}

    def levels(self, profiles, variable, qc_policy=QC_POLICY):
        """Values and the level mask of a QC policy (qc_policy_v1: profile-level rejections already applied)."""
        spec, rows = VARIABLES[variable], profiles.rows
        value = self.read(spec["value"], rows).astype(np.float64)
        if qc_policy == "finite_only":
            return value, (self.read(spec["mask"], rows) == 1) & np.isfinite(value), np.ones(len(rows), dtype=bool)
        if qc_policy != QC_POLICY:
            raise ValueError(f"unknown QC policy {qc_policy}")
        level_qc = self.read(spec["level_qc"], rows)
        mask = self.read(spec["mask"], rows)
        depth_qc = self.read("argo_depth_qc_on_glorys_depth", rows)
        ok_profile, _ = self.profile_ok(profiles, variable)
        accepted = ((mask == 1) & np.isfinite(value) & (level_qc == 1)
                    & ((depth_qc == 1) | (depth_qc == -1)) & ok_profile[:, None])
        return value, accepted, ok_profile


def partition_of(time, lat, lon):
    """partition_v1 label per profile: train/dev/test, '<split>-reserved' inside the reserved region, or 'buffer'."""
    time = pd.DatetimeIndex(pd.to_datetime(time))
    year = time.year.to_numpy()
    label = np.full(len(year), "buffer", dtype=object)
    for name, (y0, y1) in SPLIT_YEARS.items():
        label[(year >= y0) & (year <= y1)] = name
    for boundary in SPLIT_BOUNDARIES:
        near = np.abs((time - boundary).total_seconds().to_numpy()) < TIME_BUFFER_DAYS * 86400
        label[near] = "buffer"
    lat, lon = np.asarray(lat), np.asarray(lon)
    (la0, la1), (lo0, lo1) = RESERVED_REGION["lat"], RESERVED_REGION["lon"]
    b = SPATIAL_BUFFER_DEG
    inside = (lat >= la0 + b) & (lat < la1 - b) & (lon >= lo0 + b) & (lon < lo1 - b)
    near_region = (lat >= la0 - b) & (lat < la1 + b) & (lon >= lo0 - b) & (lon < lo1 + b) & ~inside
    keep = label != "buffer"
    label[keep & inside] = np.char.add(label[keep & inside].astype(str), "-reserved")
    label[keep & near_region] = "buffer"
    return label


def split_dates(split):
    """First and last calendar dates of a split whose whole day [d, d + 1) avoids every year-boundary buffer."""
    y0, y1 = SPLIT_YEARS[split]
    first, last = pd.Timestamp(year=y0, month=1, day=1), pd.Timestamp(year=y1, month=12, day=31)
    buffer, day = pd.Timedelta(days=TIME_BUFFER_DAYS), pd.Timedelta(days=1)
    for b in SPLIT_BOUNDARIES:  # buffer: |t - b| < 30 days
        if b - buffer < first + day and first < b + buffer:
            first = b + buffer
        if b - buffer < last + day and last < b + buffer:
            last = b - buffer - day
    return first, last


def region_class(lat, lon):
    """Half-open box [lat0, lat1) x [lon0, lon1): 'reserved' if wholly inside the reserved region minus its buffer,
    'outside' if it does not touch the reserved region or its buffer, otherwise 'buffer'."""
    (la0, la1), (lo0, lo1) = RESERVED_REGION["lat"], RESERVED_REGION["lon"]
    b = SPATIAL_BUFFER_DEG
    if lat[0] >= la0 + b and lat[1] <= la1 - b and lon[0] >= lo0 + b and lon[1] <= lo1 - b:
        return "reserved"
    touches = lat[0] < la1 + b and la0 - b < lat[1] and lon[0] < lo1 + b and lo0 - b < lon[1]
    return "buffer" if touches else "outside"


# Environment scopes: the partition_v1 label every reachable profile must carry. Training and development never see
# the reserved region; the spatial test reads only test-period profiles inside it. Buffers are never readable.
SCOPES = {"train": ("train", False), "dev": ("dev", False), "test": ("test", False), "test-spatial": ("test", True)}


def query_partition(scope, lat, lon, start, end):
    """Checks a whole query range (not just the profiles it happens to contain) against a scope.
    Returns the partition label, or raises ValueError with the reason."""
    split, reserved = SCOPES[scope]
    first, last = split_dates(split)
    if pd.Timestamp(start) < first or pd.Timestamp(end) > last:
        raise ValueError(f"日期超出本环境分区 {scope} 的可用范围 {first:%Y-%m-%d}～{last:%Y-%m-%d}（含年界缓冲）")
    where = region_class(lat, lon)
    if where != ("reserved" if reserved else "outside"):
        reason = {"buffer": "区域触及预留海域或其 1° 缓冲带", "reserved": "区域位于预留海域内",
                  "outside": "区域不在预留海域内"}[where]
        raise ValueError(f"{reason}，不属于本环境分区 {scope}")
    return f"{split}-reserved" if reserved else split


def profile_set_digest(profiles):
    """Provenance hash of a selected profile set: its unique keys (source file, row in file)."""
    keys = profiles.index["profile_source_file"].astype(str) + ":" + profiles.index["profile_idx"].astype(str)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()
