#!/usr/bin/env python3
"""
BioDCASE 2026 — Stage-2 Feature Builder (ARIA detection format)
================================================================

Builds species-specific population-estimation features from ARIA detection
CSVs for the BioDCASE 2026 Challenge target species.

This script reads detection CSVs produced by ``aria-inference``
(``pip install aria-inference``) and extracts a rich feature vector per
(aviary, target-species) pair.  Features include detection-count statistics,
temporal bout structure, and optionally scikit-maad acoustic indices.

The feature block is designed around three ideas:

1. **scene context** — summaries over a sampled set of all segments in an aviary
2. **target-positive acoustics** — summaries over segments where the target
   species was detected
3. **contrast features** — positive-minus-background and positive-vs-scene ratios

Additionally, five composite acoustic scores are computed as transformed
summaries of the raw index block (log-compressed means and log-ratios),
suitable for direct use in small-sample regression models without feature
explosion.

Detection CSV naming convention
--------------------------------
Participants must name their detection CSVs as::

    dev_aviary_1_detections.csv
    dev_aviary_2_detections.csv
    ...
    dev_aviary_6_detections.csv

ARIA detection format (from ``pip install aria-inference``)::

    File,Segment,Start,End,Species,Confidence,Method,Status

Usage
-----
::

    python feature_builder.py \\
        --detections-dir ./detections \\
        --output features/stage2_features.csv \\
        --audio-root /path/to/audio \\
        --device cuda --workers 4

Then run the estimator on the ARIA features::

    python estimator.py --features features/stage2_features.csv

Performance notes
-----------------
- ``--workers N``  parallelizes segment-level acoustic extraction with a
  thread pool (GIL released during I/O and scipy C-extension calls).
- ``--device cuda``  offloads spectrogram computation to GPU via torch.stft,
  matching scipy PSD normalisation so downstream maad features are unaffected.
  scikit-maad feature functions themselves remain CPU-only.
"""

import concurrent.futures
import csv
import hashlib
import json
import math
import re
import threading
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

warnings.filterwarnings("ignore")


def _stable_u32_seed(*parts: object) -> int:
    """Return a stable 32-bit seed from arbitrary parts."""
    payload = "||".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)

try:
    from maad import features as maad_features
    from maad import sound as maad_sound
    from maad import util as maad_util
    MAAD_AVAILABLE = True
except Exception:
    maad_features = None
    maad_sound = None
    maad_util = None
    MAAD_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    TORCH_AVAILABLE = False

DEFAULT_BOUT_GAP_SECONDS = 6.0
HIGH_CONF_THRESHOLD = 0.70
EMBEDDING_CLUSTER_COSINE_DISTANCE = 0.15
EMBEDDING_SAMPLE_LIMIT = 200

# ── dB floor for noise-equalised spectrograms ──
# prevents -inf from propagating through downstream band metrics
DB_FLOOR = -120.0

# ── silent-segment RMS threshold ──
# segments below this RMS are treated as silence and skipped
SILENCE_RMS_THRESHOLD = 1e-6

DEFAULT_PATHS = {
    "aviary_csv": Path(__file__).parent / "ground_truth.csv",
    "aviary_config": Path(__file__).parent / "aviary_config.json",
    "detections_dir": Path(__file__).parent / "detections",
    "output_dir": Path(__file__).parent / "features",
    "embeddings_dir": None,
}


def load_aviary_day_mappings(config_path: str) -> Dict[str, Dict[str, str]]:
    """Load aviary_config.json and return per-aviary day mappings.

    Returns ``{aviary_id: {dN: "YYYY-MM-DD"}}`` — e.g.
    ``{"dev_aviary_1": {"d1": "2025-01-01", "d2": "2025-01-02", ...}}``.

    The mappings are used by ``parse_filename_timestamp`` to resolve
    the BioDCASE filename format ``rec_d1_19_05_02.500000.wav`` into a
    full datetime.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        aviary_id: info.get("day_mapping", {})
        for aviary_id, info in config.items()
    }

TARGET_SPECIES = {
    "greater flamingo",
    "red billed quelea",
    "hadada ibis"
}


CSV_TO_AVIARY = {
    "dev_aviary_1": ["dev_aviary_1"],
    "dev_aviary_2": ["dev_aviary_2"],
    "dev_aviary_3": ["dev_aviary_3"],
    "dev_aviary_4": ["dev_aviary_4"],
    "dev_aviary_5": ["dev_aviary_5"],
    "dev_aviary_6": ["dev_aviary_6"],
}

# Maps session-specific aviary names to the ground-truth aviary name.
# Not needed for BioDCASE challenge aviaries (each has a unique ID).
AVIARY_GT_ALIAS = {}

CSV_TO_EMBEDDING_PREFIX = {
    "dev_aviary_1": "dev_aviary_1",
    "dev_aviary_2": "dev_aviary_2",
    "dev_aviary_3": "dev_aviary_3",
    "dev_aviary_4": "dev_aviary_4",
    "dev_aviary_5": "dev_aviary_5",
    "dev_aviary_6": "dev_aviary_6",
}

# Some species have multiple common names in the dataset.  We want to unify these
SPECIES_ALIASES = {
    "gray crowned crane": "grey crowned crane",
    "grey crowned crane": "gray crowned crane",
}

# editable engineering defaults for target-band descriptors.
# Widened based on literature review — these are conservative "safe" bands
# that capture the core energy plus harmonics.  Per-species empirical
# tuning from spectrograms is a future refinement.
DEFAULT_SPECIES_BANDS = {
    "greater flamingo": (300.0, 5000.0),
    "red billed quelea": (500.0, 6500.0),
    "hadada ibis": (300.0, 3000.0),
}

# Narrow "core energy" bands — the original literature-derived ranges before
# widening.  The adaptive band selector compares narrow vs. wide and picks
# whichever produces stronger positive-minus-background contrast per aviary.
NARROW_SPECIES_BANDS = {
    "greater flamingo": (300.0, 2500.0),
    "red billed quelea": (2000.0, 8000.0),
    "hadada ibis": (300.0, 1800.0),
}

MAAD_COLUMN_MAP = {
    "LEQt": "maad_leqt",
    "Ht": "maad_ht",
    "ACTtFraction": "maad_acttfraction",
    "EVNtCount": "maad_evntcount",
    "ACI": "maad_aci",
    "NDSI": "maad_ndsi",
    "BI": "maad_bi",
    "ADI": "maad_adi",
    "AEI": "maad_aei",
    "Hf": "maad_hf",
    "LFC": "maad_lfc",
    "MFC": "maad_mfc",
    "HFC": "maad_hfc",
    "ACTspFract": "maad_actspfract",
    "EVNspCount": "maad_evnspcount",
    "nROI": "maad_nroi",
    "aROI": "maad_aroi",
    "TFSD": "maad_tfsd",
    "AGI": "maad_agi",
}

MAAD_BAND_FEATURES = [
    "maad_target_band_power_frac",
    "maad_target_band_cover",
    "maad_target_band_activity_frac",
]

MAAD_BASE_FEATURES = list(MAAD_COLUMN_MAP.values()) + MAAD_BAND_FEATURES
MAAD_SAMPLE_STATS = [
    "all_mean",
    "all_std",
    "pos_mean",
    "pos_median",
    "pos_p90",
    "bg_mean",
    "bg_median",
    "pos_minus_bg",
    "pos_ratio_bg",
    "pos_minus_all",
    "pos_ratio_all",
]

MAAD_AGG_COLUMNS = [
    "acoustic_all_window_count",
    "acoustic_positive_window_count",
    "acoustic_background_window_count",
    "acoustic_windows_missing",
    "acoustic_resolved_file_fraction",
    "target_band_fmin_hz",
    "target_band_fmax_hz",
] + [f"{feat}__{stat}" for feat in MAAD_BASE_FEATURES for stat in MAAD_SAMPLE_STATS]

# ── composite acoustic features ──
# These are dimensionless summary scores derived from the full acoustic index
# block.  They are designed to be directly usable by the small-sample
# regression models without inflating the feature space.
#
# - scene_complexity:   how acoustically complex the whole aviary recording is
# - target_intensity:   acoustic complexity in target-positive windows only
# - target_contrast_ratio:  ratio-form contrast (pos/bg) averaged over key indices
# - target_band_contrast:   contrast for the species-specific frequency band
# - event_density_contrast: ratio-form contrast for event-count indices
COMPOSITE_ACOUSTIC_COLUMNS = [
    "acoustic_scene_complexity",
    "acoustic_target_intensity",
    "acoustic_target_contrast_ratio",
    "acoustic_target_band_contrast",
    "acoustic_event_density_contrast",
]

# ── flock-calling features ──
# These quantify how "flock-like" the calling pattern is in each aviary.
# High flock index → many birds calling synchronously → raw CWR undercounts.
FLOCK_COLUMNS = [
    "flock_energy_stability",       # 1/CV of LEQt across positive windows
    "flock_event_suppression",      # nROI ratio (pos/bg); < 1 = events merging
    "flock_spectral_persistence",   # 1/CV of spectral activity across positive windows
    "flock_bg_bleed",               # target-band activity in "background" windows
    "flock_band_energy_stability",  # 1/CV of target-band power across positive windows
    "flock_index",                  # composite flock score (geometric mean)
    "flock_corrected_cwr",          # CWR adjusted upward by flock index
]

# ── adaptive band features ──
# Instead of using a fixed band, we compute band metrics at both the narrow
# (core energy) and wide (full structure) bands, then pick whichever yields
# stronger positive-minus-background contrast for this aviary.
ADAPTIVE_BAND_COLUMNS = [
    "adaptive_band_fmin_hz",
    "adaptive_band_fmax_hz",
    "adaptive_band_source",         # 0 = narrow, 1 = wide
    "adaptive_band_power_frac_contrast",   # pos_minus_bg at the selected band
    "adaptive_band_cover_contrast",
    "adaptive_band_activity_contrast",
]


@dataclass
class AviaryGroundTruth:
    aviary_name: str
    species: Dict[str, int]
    original_names: Dict[str, str]
    scientific_names: Dict[str, str]


@dataclass
class TargetSpeciesFeatures:
    payload: Dict[str, object]


FEATURE_COLUMNS = [
    "recording_hours",
    "coverage_hours",
    "n_total_segments",
    "total_files",
    "total_detections",
    "detection_rate_per_hour",
    "confidence_weighted_rate",
    "mean_confidence",
    "std_confidence",
    "bout_count",
    "bout_rate_per_hour",
    "total_bout_duration_sec",
    "mean_bout_duration_sec",
    "max_bout_duration_sec",
    "mean_segments_per_bout",
    "max_segments_per_bout",
    "active_files",
    "active_file_fraction",
    "active_hours",
    "temporal_spread",
    "positive_segment_seconds",
    "positive_audio_fraction",
    "bout_span_fraction",
    "longest_positive_run_segments",
    "longest_positive_run_seconds",
    "peak_hour_share_of_positive_segments",
    "mean_inter_bout_gap_sec",
    "median_inter_bout_gap_sec",
    "min_inter_bout_gap_sec",
    "mean_species_labels_per_positive_segment",
    "max_species_labels_per_positive_segment",
    "fraction_positive_segments_with_2plus_labels",
    "fraction_positive_segments_with_3plus_labels",
    "mean_other_species_per_positive_segment",
    "mean_highconf_other_species_per_positive_segment",
    "overlap_segments_any",
    "overlap_fraction_any",
    "overlap_segments_highconf",
    "overlap_fraction_highconf",
    "mean_target_conf_margin_vs_best_other",
    "embedding_count",
    "embedding_mean_norm",
    "embedding_var_mean",
    "embedding_first_pc_ratio",
    "embedding_mean_pairwise_distance",
    "embedding_median_nn_distance",
    "embedding_cluster_count_proxy",
    *MAAD_AGG_COLUMNS,
    *COMPOSITE_ACOUSTIC_COLUMNS,
    *FLOCK_COLUMNS,
    *ADAPTIVE_BAND_COLUMNS,
]


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def normalize_species_name(name: str) -> str:
    if "_" in name:
        name = name.split("_", 1)[1]
    return name.lower().replace("-", " ").replace("_", " ").strip()


def parse_aviary_csv(csv_path: str) -> Dict[str, AviaryGroundTruth]:
    """Parse ground_truth.csv with columns: aviary_id,common_name,scientific_name,count,is_target"""
    aviaries = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aviary_name = row.get("aviary_id", "").strip()
            common_name = row.get("common_name", "").strip()
            scientific_name = row.get("scientific_name", "").strip()
            if not aviary_name or not common_name:
                continue
            try:
                count = int(row.get("count", "0").strip())
            except ValueError:
                continue

            if aviary_name not in aviaries:
                aviaries[aviary_name] = AviaryGroundTruth(
                    aviary_name=aviary_name,
                    species={},
                    original_names={},
                    scientific_names={},
                )

            norm = normalize_species_name(common_name)
            aviaries[aviary_name].species[norm] = count
            aviaries[aviary_name].original_names[norm] = common_name
            aviaries[aviary_name].scientific_names[norm] = scientific_name
    return aviaries


def parse_filename_timestamp(filename: str, day_mapping: Optional[Dict[str, str]] = None) -> Optional[datetime]:
    """Extract a datetime from a recording filename.

    Supports two naming conventions:

    1. **Full-date format** (legacy / original recordings)::

           er_file_2025_04_15_19_05_02.500000.wav
                    YYYY MM DD HH MM SS.ffffff

    2. **BioDCASE day-mapped format** (anonymised dataset)::

           rec_d1_19_05_02.500000.wav
               dN HH MM SS.ffffff

       Requires *day_mapping* — a dict like ``{"d1": "2025-01-01", ...}`` —
       loaded from ``aviary_config.json``.
    """
    # ── try BioDCASE day-mapped format first ──
    if day_mapping:
        match = re.search(r"(d\d+)_(\d{1,2})_(\d{2})_(\d{2})(?:\.(\d{1,6}))?", filename)
        if match:
            day_key = match.group(1)
            date_str = day_mapping.get(day_key)
            if date_str:
                try:
                    hour, minute, second = int(match.group(2)), int(match.group(3)), int(match.group(4))
                    frac = match.group(5)
                    microsecond = int(frac.ljust(6, "0")) if frac else 0
                    base_date = datetime.strptime(date_str, "%Y-%m-%d")
                    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
                except ValueError:
                    pass

    # ── fall back to full-date format ──
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:\.(\d{1,6}))?", filename)
    if not match:
        return None
    try:
        year, month, day, hour, minute, second = map(int, match.groups()[:6])
        frac = match.group(7)
        microsecond = int(frac.ljust(6, "0")) if frac else 0
        return datetime(year, month, day, hour, minute, second, microsecond)
    except ValueError:
        return None


def parse_aria_inference_csv(csv_path: str, day_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[Dict], List[Dict]]:
    segment_map = {}
    detections = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Status", "") != "success":
                continue
            file_name = row.get("File", "")
            segment_id = row.get("Segment", "0")
            try:
                start = float(row.get("Start", 0))
                end = float(row.get("End", 0))
            except ValueError:
                continue
            duration = max(end - start, 0.0)
            timestamp = parse_filename_timestamp(file_name, day_mapping=day_mapping)
            abs_start = timestamp + timedelta(seconds=start) if timestamp else None
            abs_end = timestamp + timedelta(seconds=end) if timestamp else None
            seg_key = (file_name, str(segment_id), start, end)
            if seg_key not in segment_map:
                segment_map[seg_key] = {
                    "file": file_name,
                    "file_base": Path(file_name).name,
                    "segment": str(segment_id),
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "timestamp": timestamp,
                    "abs_start": abs_start,
                    "abs_end": abs_end,
                    "key": seg_key,
                }
            species_raw = row.get("Species", "")
            if not species_raw or species_raw == "NO_PREDICTION":
                continue
            try:
                confidence = float(row.get("Confidence", 0))
            except ValueError:
                continue
            detections.append({
                "file": file_name,
                "file_base": Path(file_name).name,
                "segment": str(segment_id),
                "species_raw": species_raw,
                "species_normalized": normalize_species_name(species_raw),
                "confidence": confidence,
                "start": start,
                "end": end,
                "timestamp": timestamp,
                "abs_start": abs_start,
                "abs_end": abs_end,
                "segment_key": seg_key,
            })
    return list(segment_map.values()), detections


def compute_audio_hours(segments: List[Dict]) -> float:
    return max(sum(max(seg["duration"], 0.0) for seg in segments) / 3600.0, 1e-8)


def compute_coverage_hours(segments: List[Dict]) -> float:
    timestamps = [seg["timestamp"] for seg in segments if seg["timestamp"] is not None]
    if len(timestamps) < 2:
        return 0.0
    return max((max(timestamps) - min(timestamps)).total_seconds() / 3600.0, 0.0)


def count_total_unique_hours(segments: List[Dict]) -> int:
    unique_hours = set()
    for seg in segments:
        ts = seg.get("timestamp")
        if ts is not None:
            unique_hours.add(ts.replace(minute=0, second=0, microsecond=0))
    return max(len(unique_hours), 1)


def make_segment_detection_index(detections: List[Dict]) -> Dict[Tuple, List[Dict]]:
    idx = defaultdict(list)
    for det in detections:
        idx[det["segment_key"]].append(det)
    return idx


def build_species_positive_segments(matched_dets: List[Dict], segment_map: Dict[Tuple, Dict]) -> List[Dict]:
    seen = set()
    positive_segments = []
    for det in matched_dets:
        key = det["segment_key"]
        if key in seen or key not in segment_map:
            continue
        seen.add(key)
        positive_segments.append(segment_map[key])
    positive_segments.sort(key=lambda s: (s["abs_start"] if s["abs_start"] is not None else datetime.max, s["file"], s["segment"]))
    return positive_segments


def build_bouts(positive_segments: List[Dict], gap_seconds: float) -> List[Dict]:
    if not positive_segments:
        return []
    bouts = []
    current = {"start": positive_segments[0]["abs_start"], "end": positive_segments[0]["abs_end"], "segments": [positive_segments[0]]}
    for seg in positive_segments[1:]:
        if current["end"] is None or seg["abs_start"] is None:
            gap = float("inf")
        else:
            gap = (seg["abs_start"] - current["end"]).total_seconds()
        if gap <= gap_seconds:
            current["segments"].append(seg)
            if current["end"] is None or (seg["abs_end"] is not None and seg["abs_end"] > current["end"]):
                current["end"] = seg["abs_end"]
        else:
            bouts.append(current)
            current = {"start": seg["abs_start"], "end": seg["abs_end"], "segments": [seg]}
    bouts.append(current)
    for bout in bouts:
        if bout["start"] is not None and bout["end"] is not None:
            bout["duration_sec"] = max((bout["end"] - bout["start"]).total_seconds(), 0.0)
        else:
            bout["duration_sec"] = float(sum(seg["duration"] for seg in bout["segments"]))
        bout["n_segments"] = len(bout["segments"])
    return bouts


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else default


def _safe_median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.median(vals)) if vals else default


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.std(vals)) if vals else default


def _safe_p90(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.percentile(vals, 90)) if vals else default


def _safe_ratio(a: float, b: float, default: float = float("nan")) -> float:
    if b is None or np.isnan(b) or abs(b) < 1e-12:
        return default
    if a is None or np.isnan(a):
        return default
    return float(a / b)


def _find_npz_for_prefix(embeddings_dir: Path, prefix: str) -> Optional[Path]:
    preferred = list(embeddings_dir.glob(f"{prefix}*1536dim*.npz"))
    if preferred:
        return sorted(preferred)[0]
    any_npz = list(embeddings_dir.glob(f"{prefix}*.npz"))
    return sorted(any_npz)[0] if any_npz else None


# ═══════════════════════════════════════════════════════════════════════════
# Embedding helpers
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    def __init__(self, embeddings_dir: Optional[str]):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.cache = {}

    def get_index(self, csv_basename: str) -> Optional[Dict[Tuple[str, str], np.ndarray]]:
        if self.embeddings_dir is None:
            return None
        prefix = CSV_TO_EMBEDDING_PREFIX.get(csv_basename)
        if not prefix:
            return None
        if prefix in self.cache:
            return self.cache[prefix]

        npz_path = _find_npz_for_prefix(self.embeddings_dir, prefix)
        if npz_path is None:
            self.cache[prefix] = None
            return None

        data = np.load(npz_path, allow_pickle=True)
        if "embeddings" not in data or "file_names" not in data or "segment_indices" not in data:
            self.cache[prefix] = None
            return None

        embeddings = data["embeddings"]
        file_names = data["file_names"]
        segment_indices = data["segment_indices"]

        index = {}
        for emb, fn, seg_idx in zip(embeddings, file_names, segment_indices):
            file_base = Path(str(fn)).name
            index[(file_base, str(int(seg_idx)))] = np.asarray(emb, dtype=np.float64)

        self.cache[prefix] = index
        return index


def _sample_embeddings(X: np.ndarray, limit: int = EMBEDDING_SAMPLE_LIMIT) -> np.ndarray:
    if X.shape[0] <= limit:
        return X
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=limit, replace=False)
    return X[idx]


def _embedding_feature_summary(embeddings: List[np.ndarray]) -> Dict[str, float]:
    if not embeddings:
        return {
            "embedding_count": 0,
            "embedding_mean_norm": float("nan"),
            "embedding_var_mean": float("nan"),
            "embedding_first_pc_ratio": float("nan"),
            "embedding_mean_pairwise_distance": float("nan"),
            "embedding_median_nn_distance": float("nan"),
            "embedding_cluster_count_proxy": float("nan"),
        }

    X = np.asarray(embeddings, dtype=np.float64)
    X = _sample_embeddings(X)
    n = X.shape[0]

    norms = np.linalg.norm(X, axis=1)
    centered = X - np.mean(X, axis=0, keepdims=True)
    var_mean = float(np.mean(np.var(centered, axis=0)))

    if n >= 2:
        try:
            _, svals, _ = np.linalg.svd(centered, full_matrices=False)
            pc_ratio = float((svals[0] ** 2) / np.sum(svals ** 2)) if np.sum(svals ** 2) > 0 else float("nan")
        except np.linalg.LinAlgError:
            pc_ratio = float("nan")
    else:
        pc_ratio = float("nan")

    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = Xn @ Xn.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.nan)
    pairwise = dist[~np.isnan(dist)]
    mean_pairwise = float(np.mean(pairwise)) if pairwise.size else float("nan")
    nn = np.nanmin(dist, axis=1) if n > 1 else np.array([float("nan")])
    median_nn = float(np.nanmedian(nn)) if np.any(~np.isnan(nn)) else float("nan")

    order = list(range(n))
    exemplars = []
    for i in order:
        if not exemplars:
            exemplars.append(i)
            continue
        d_to_exemplars = [dist[i, j] for j in exemplars]
        if np.nanmin(d_to_exemplars) > EMBEDDING_CLUSTER_COSINE_DISTANCE:
            exemplars.append(i)
    cluster_proxy = float(len(exemplars))

    return {
        "embedding_count": int(n),
        "embedding_mean_norm": float(np.mean(norms)),
        "embedding_var_mean": var_mean,
        "embedding_first_pc_ratio": pc_ratio,
        "embedding_mean_pairwise_distance": mean_pairwise,
        "embedding_median_nn_distance": median_nn,
        "embedding_cluster_count_proxy": cluster_proxy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sampling helper
# ═══════════════════════════════════════════════════════════════════════════

def _deterministic_sample(items: Sequence[Tuple], max_n: Optional[int], seed: int) -> List[Tuple]:
    items = list(items)
    if max_n is None or max_n <= 0 or len(items) <= max_n:
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=max_n, replace=False))
    return [items[i] for i in idx]


# ═══════════════════════════════════════════════════════════════════════════
# Audio path resolver
# ═══════════════════════════════════════════════════════════════════════════

class AudioPathResolver:
    """Resolves detection-CSV file references to actual WAV paths on disk.

    On construction, walks every audio root **once** and builds a
    ``basename → [paths]`` index.  All subsequent ``resolve()`` calls
    are O(1) dict lookups.

    When multiple files share the same basename (common with the BioDCASE
    naming scheme ``rec_d1_HH_MM_SS.wav`` across aviaries), an optional
    ``aviary_hint`` narrows the lookup to paths containing that string
    (e.g. ``"dev_aviary_1"``).
    """

    def __init__(self, audio_roots: Sequence[str]):
        self.audio_roots = [Path(p) for p in audio_roots if p]
        # Pre-index: walk every root once and map basename → [paths].
        self._index: Dict[str, List[Path]] = {}
        n_files = 0
        for root in self.audio_roots:
            if not root.is_dir():
                continue
            print(f"   indexing audio root: {root} ...", end=" ", flush=True)
            count = 0
            for ext in ("*.wav", "*.WAV"):
                for p in root.rglob(ext):
                    bn = p.name
                    if bn not in self._index:
                        self._index[bn] = [p]
                    else:
                        # avoid exact duplicates (e.g. from *.wav and *.WAV overlap)
                        if p not in self._index[bn]:
                            self._index[bn].append(p)
                    count += 1
            n_files += count
            print(f"{count:,d} files indexed ({len(self._index):,d} unique basenames)")
        if n_files:
            print(f"  ✓ audio index ready: {n_files:,d} total files, {len(self._index):,d} unique basenames across {len(self.audio_roots)} root(s)")
        else:
            print("    no WAV files found in any audio root")

    def resolve(self, file_ref: str, aviary_hint: Optional[str] = None) -> Optional[Path]:
        if not file_ref:
            return None
        # fast path: direct absolute/relative path exists
        direct = Path(file_ref)
        if direct.exists():
            return direct
        # fast path: root + full ref or root + basename
        for root in self.audio_roots:
            candidate = root / file_ref
            if candidate.exists():
                return candidate
            candidate = root / direct.name
            if candidate.exists():
                return candidate
        # O(1) lookup in pre-built index
        candidates = self._index.get(direct.name)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple matches — use aviary_hint to disambiguate
        if aviary_hint:
            for p in candidates:
                if aviary_hint in p.parts:
                    return p
        # Fallback: return first match
        return candidates[0]


# ═══════════════════════════════════════════════════════════════════════════
# GPU spectrogram helper
# ═══════════════════════════════════════════════════════════════════════════

def _torch_psd_spectrogram(
    y: np.ndarray,
    sr: int,
    nperseg: int,
    noverlap: int,
    device: "torch.device",
    window_tensor: "torch.Tensor",
    window_sq_sum: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """GPU-accelerated PSD spectrogram matching scipy.signal.spectrogram output.

    Uses torch.stft on the specified device, then normalises to one-sided
    power-spectral-density scaling identical to
    ``scipy.signal.spectrogram(mode='psd', scaling='density')``.

    Parameters match ``maad_sound.spectrogram`` return format:
    ``(Sxx_power, tn, fn, extent)``.
    """
    hop = nperseg - noverlap
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.float32)

    stft_out = torch.stft(
        y_t,
        n_fft=nperseg,
        hop_length=hop,
        win_length=nperseg,
        window=window_tensor,
        return_complex=True,
        onesided=True,
        center=False,
    )
    # stft_out shape: (n_fft//2 + 1, n_frames)
    Sxx = (stft_out.real ** 2 + stft_out.imag ** 2).cpu().numpy()

    # PSD normalisation: Sxx / (fs * sum(window^2))
    Sxx /= (sr * window_sq_sum)
    # one-sided doubling (skip DC at [0] and Nyquist at [-1])
    if Sxx.shape[0] > 2:
        Sxx[1:-1, :] *= 2.0

    fn = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    tn = (np.arange(Sxx.shape[1]) * hop + nperseg // 2) / sr
    ext = [float(tn[0]), float(tn[-1]), float(fn[0]), float(fn[-1])]
    return Sxx, tn, fn, ext


# ═══════════════════════════════════════════════════════════════════════════
# Acoustic index extractor
# ═══════════════════════════════════════════════════════════════════════════

class AcousticIndexExtractor:
    def __init__(
        self,
        audio_roots: Sequence[str],
        species_bands: Dict[str, Tuple[float, float]],
        narrow_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        max_sample_rate: int = 32000,
        spectrogram_nperseg: int = 1024,
        spectrogram_noverlap: int = 512,
        activity_db_threshold: float = 3.0,
        event_db_threshold: float = 3.0,
        reject_duration: float = 0.10,
        adi_db_threshold: float = -50.0,
        aei_db_threshold: float = -50.0,
        min_segment_duration: float = 0.25,
        device: str = "cpu",
        max_workers: int = 1,
    ):
        if not MAAD_AVAILABLE:
            raise RuntimeError(
                "scikit-maad is not installed. Install it with `pip install scikit-maad` before using acoustic indices."
            )
        self.resolver = AudioPathResolver(audio_roots)
        self.species_bands = dict(species_bands)
        self.narrow_bands = dict(narrow_bands) if narrow_bands else {}
        # Compute band metrics at BOTH narrow and wide bands per segment.
        # The adaptive band selector picks the best one at aggregation time.
        all_bands = set(tuple(v) for v in species_bands.values())
        if narrow_bands:
            all_bands |= set(tuple(v) for v in narrow_bands.values())
        self.unique_bands = sorted(all_bands)
        self.max_sample_rate = int(max_sample_rate)
        self.spectrogram_nperseg = int(spectrogram_nperseg)
        self.spectrogram_noverlap = int(spectrogram_noverlap)
        self.activity_db_threshold = float(activity_db_threshold)
        self.event_db_threshold = float(event_db_threshold)
        self.reject_duration = float(reject_duration)
        self.adi_db_threshold = float(adi_db_threshold)
        self.aei_db_threshold = float(aei_db_threshold)
        self.min_segment_duration = float(min_segment_duration)
        self.max_workers = max(int(max_workers), 1)

        # ── thread-safe segment cache ──
        self.segment_cache: Dict[Tuple, Optional[Dict[str, float]]] = {}
        self._cache_lock = threading.Lock()

        # ── GPU setup ──
        self.device_name = device
        self.use_gpu = device != "cpu" and TORCH_AVAILABLE
        self._torch_device = None
        self._torch_window = None
        self._window_sq_sum = 0.0
        if self.use_gpu:
            self._torch_device = torch.device(device)
            self._torch_window = torch.hann_window(spectrogram_nperseg, device=self._torch_device)
            self._window_sq_sum = float((self._torch_window ** 2).sum().cpu().item())
            # lock for serialising GPU stft calls across threads
            self._gpu_lock = threading.Lock()

    def _resample_if_needed(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if sr <= self.max_sample_rate:
            return y.astype(np.float32, copy=False), sr
        gcd = math.gcd(sr, self.max_sample_rate)
        up = self.max_sample_rate // gcd
        down = sr // gcd
        y_rs = signal.resample_poly(y, up, down).astype(np.float32, copy=False)
        return y_rs, self.max_sample_rate

    def _read_segment(self, path: Path, start_sec: float, end_sec: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
        try:
            info = sf.info(str(path))
            sr = int(info.samplerate)
            start_frame = max(int(round(start_sec * sr)), 0)
            end_frame = max(int(round(end_sec * sr)), start_frame + 1)
            y = sf.read(str(path), start=start_frame, stop=end_frame, dtype="float32", always_2d=False)[0]
        except Exception:
            return None, None

        if y is None:
            return None, None
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        if y.size < max(int(self.min_segment_duration * sr), 16):
            return None, None
        y, sr = self._resample_if_needed(y, sr)

        # ── reject near-silent segments ──
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            return None, None

        return y, sr

    def _compute_spectrogram(
        self, y: np.ndarray, sr: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
        """Compute PSD spectrogram via GPU (torch) or CPU (maad)."""
        if self.use_gpu:
            try:
                with self._gpu_lock:
                    return _torch_psd_spectrogram(
                        y, sr,
                        nperseg=self.spectrogram_nperseg,
                        noverlap=self.spectrogram_noverlap,
                        device=self._torch_device,
                        window_tensor=self._torch_window,
                        window_sq_sum=self._window_sq_sum,
                    )
            except Exception:
                # fall back to CPU on any GPU error
                pass
        return maad_sound.spectrogram(
            y, sr,
            nperseg=self.spectrogram_nperseg,
            noverlap=self.spectrogram_noverlap,
            detrend=False,
        )

    @staticmethod
    def _extract_first_row(df) -> Dict[str, float]:
        if df is None or len(df) == 0:
            return {}
        row = df.iloc[0]
        out = {}
        for key, val in row.items():
            try:
                out[str(key)] = float(val)
            except Exception:
                continue
        return out

    def _band_key(self, band: Tuple[float, float]) -> str:
        return f"band_{int(round(band[0]))}_{int(round(band[1]))}"

    def _band_metrics(self, Sxx_power: np.ndarray, Sxx_dB_no_noise: np.ndarray, fn: np.ndarray, band: Tuple[float, float]) -> Dict[str, float]:
        fmin, fmax = band
        mask = (fn >= fmin) & (fn < fmax)
        if not np.any(mask):
            return {
                "power_frac": float("nan"),
                "cover": float("nan"),
                "activity_frac": float("nan"),
            }
        band_power = float(np.sum(Sxx_power[mask, :]))
        total_power = float(np.sum(Sxx_power))
        active = Sxx_dB_no_noise[mask, :] > self.activity_db_threshold
        return {
            "power_frac": band_power / total_power if total_power > 0 else float("nan"),
            "cover": float(np.mean(active)) if active.size else float("nan"),
            "activity_frac": float(np.mean(np.any(active, axis=0))) if active.shape[1] > 0 else float("nan"),
        }

    def _compute_segment_features(self, segment: Dict, aviary_hint: Optional[str] = None) -> Optional[Dict[str, float]]:
        audio_path = self.resolver.resolve(segment["file"], aviary_hint=aviary_hint)
        if audio_path is None:
            return None
        y, sr = self._read_segment(audio_path, float(segment["start"]), float(segment["end"]))
        if y is None or sr is None:
            return None

        try:
            spec_result = self._compute_spectrogram(y, sr)
            if spec_result is None:
                return None
            Sxx_power, tn, fn, _ = spec_result

            if Sxx_power.size == 0 or len(tn) < 2 or len(fn) < 2:
                return None
            temporal_df = maad_features.all_temporal_alpha_indices(y, sr, display=False)

            # ── remove_rain=False ──
            # Zoo aviaries may contain water features (fountains, misters) whose
            # spectral signature triggers maad's rain-removal heuristic.  Keeping
            # rain removal off avoids asymmetric suppression across aviaries.
            spectral_df, _ = maad_features.all_spectral_alpha_indices(
                Sxx_power,
                tn,
                fn,
                flim_low=[0, 1000],
                flim_mid=[1000, 10000],
                flim_hi=[10000, 20000],
                display=False,
                dB_threshold=self.activity_db_threshold,
                rejectDuration=self.reject_duration,
                ADI_dB_threshold=self.adi_db_threshold,
                AEI_dB_threshold=self.aei_db_threshold,
                remove_rain=False,
            )
            Sxx_no_noise = maad_sound.median_equalizer(Sxx_power)
            Sxx_dB_no_noise = maad_util.power2dB(Sxx_no_noise)

            # ──  clamp dB floor ──
            # median_equalizer can produce zeros → power2dB yields -inf.
            # Clamping to DB_FLOOR prevents -inf from propagating into
            # downstream band_metrics thresholding.
            Sxx_dB_no_noise = np.maximum(Sxx_dB_no_noise, DB_FLOOR)

        except Exception:
            return None

        features = {}
        row_temporal = self._extract_first_row(temporal_df)
        row_spectral = self._extract_first_row(spectral_df)
        for source_name, out_name in MAAD_COLUMN_MAP.items():
            if source_name in row_temporal:
                features[out_name] = row_temporal[source_name]
            elif source_name in row_spectral:
                features[out_name] = row_spectral[source_name]
            else:
                features[out_name] = float("nan")

        for band in self.unique_bands:
            band_key = self._band_key(band)
            vals = self._band_metrics(Sxx_power, Sxx_dB_no_noise, fn, band)
            for metric_name, val in vals.items():
                features[f"{band_key}__{metric_name}"] = val

        return features

    def get_segment_features(self, segment: Dict, aviary_hint: Optional[str] = None) -> Optional[Dict[str, float]]:
        # Include aviary_hint in cache key to avoid cross-aviary collisions
        # when different aviaries share the same filename
        cache_key = (segment["key"], aviary_hint)
        with self._cache_lock:
            if cache_key in self.segment_cache:
                return self.segment_cache[cache_key]
        result = self._compute_segment_features(segment, aviary_hint=aviary_hint)
        with self._cache_lock:
            self.segment_cache[cache_key] = result
        return result

    def batch_compute_features(
        self, segments: List[Dict], label: str = "", aviary_hint: Optional[str] = None,
    ) -> Tuple[Dict[Tuple, Dict[str, float]], int, int]:
        """Compute features for a list of segments, optionally in parallel.

        Returns ``(feature_map, n_resolved, n_total)`` where *feature_map*
        maps segment keys to feature dicts (missing / failed segments are
        absent), *n_resolved* is how many audio files were successfully read,
        and *n_total* is the total number of unique segments requested.

        Parameters
        ----------
        aviary_hint : str, optional
            Aviary identifier (e.g. ``"dev_aviary_1"``) used to disambiguate
            audio files when multiple aviaries share the same basename.
        """
        seg_feature_map: Dict[Tuple, Dict[str, float]] = {}
        n_total = len(segments)
        missing = 0

        # inner progress bar so the user sees movement within an aviary
        seg_iter = segments
        use_inner_bar = tqdm is not None and n_total >= 20
        if use_inner_bar:
            seg_iter = tqdm(
                segments,
                desc=f"    acoustic [{label}]" if label else "    acoustic segments",
                unit="seg",
                leave=False,
                dynamic_ncols=True,
            )

        if self.max_workers <= 1:
            # sequential path
            for seg in seg_iter:
                feats = self.get_segment_features(seg, aviary_hint=aviary_hint)
                if feats is None:
                    missing += 1
                else:
                    seg_feature_map[seg["key"]] = feats
        else:
            # threaded parallel path
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_key = {
                    pool.submit(self.get_segment_features, seg, aviary_hint): seg["key"]
                    for seg in segments
                }
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        feats = future.result()
                    except Exception:
                        feats = None
                    if feats is None:
                        missing += 1
                    else:
                        seg_feature_map[key] = feats
                    completed += 1
                    if use_inner_bar:
                        seg_iter.update(1)

        if use_inner_bar:
            seg_iter.close()

        return seg_feature_map, n_total - missing, n_total


# ═══════════════════════════════════════════════════════════════════════════
# Acoustic aggregation
# ═══════════════════════════════════════════════════════════════════════════

def _band_key_from_band(band: Tuple[float, float]) -> str:
    return f"band_{int(round(band[0]))}_{int(round(band[1]))}"


def _attach_target_band_columns(rows: List[Dict[str, float]], band: Tuple[float, float]) -> List[Dict[str, float]]:
    band_key = _band_key_from_band(band)
    renamed_rows = []
    for row in rows:
        item = dict(row)
        item["maad_target_band_power_frac"] = row.get(f"{band_key}__power_frac", float("nan"))
        item["maad_target_band_cover"] = row.get(f"{band_key}__cover", float("nan"))
        item["maad_target_band_activity_frac"] = row.get(f"{band_key}__activity_frac", float("nan"))
        renamed_rows.append(item)
    return renamed_rows


def _aggregate_acoustic_rows(rows: List[Dict[str, float]], band: Tuple[float, float]) -> Dict[str, float]:
    out = {}
    renamed_rows = _attach_target_band_columns(rows, band)

    for feat in MAAD_BASE_FEATURES:
        vals = [r.get(feat, float("nan")) for r in renamed_rows]
        vals = [float(v) for v in vals if v is not None and not np.isnan(v)]
        out[f"{feat}__mean"] = _safe_mean(vals, float("nan"))
        out[f"{feat}__median"] = _safe_median(vals, float("nan"))
        out[f"{feat}__std"] = _safe_std(vals, float("nan"))
        out[f"{feat}__p90"] = _safe_p90(vals, float("nan"))
    return out


def _build_acoustic_feature_block(
    species_norm: str,
    aviary_name: str,
    segments: List[Dict],
    positive_segments: List[Dict],
    acoustic_extractor: Optional[AcousticIndexExtractor],
    max_all_segments: int,
    max_positive_segments: int,
    max_background_segments: int,
    cwr: float = 0.0,
    occupancy: float = 0.0,
) -> Dict[str, float]:
    if acoustic_extractor is None:
        return {col: float("nan") for col in MAAD_AGG_COLUMNS + COMPOSITE_ACOUSTIC_COLUMNS + FLOCK_COLUMNS + ADAPTIVE_BAND_COLUMNS}

    all_segments_sorted = sorted(segments, key=lambda s: (s["file"], float(s["start"]), s["segment"]))
    pos_segments_sorted = sorted(positive_segments, key=lambda s: (s["file"], float(s["start"]), s["segment"]))
    pos_keys = {seg["key"] for seg in pos_segments_sorted}

    # ──  background definition ──
    # "Background" means segments where the *current target species* was NOT
    # detected.  Other species (including other target species) may still be
    # present in these segments.  This is intentional: the contrast features
    # measure "how the scene changes when THIS species is acoustically active"
    # which requires using the species-absent scene as the reference.
    bg_segments_sorted = [seg for seg in all_segments_sorted if seg["key"] not in pos_keys]

    seed_base = _stable_u32_seed(aviary_name, species_norm)
    all_sample = _deterministic_sample([seg["key"] for seg in all_segments_sorted], max_all_segments, seed_base + 1)
    pos_sample = _deterministic_sample([seg["key"] for seg in pos_segments_sorted], max_positive_segments, seed_base + 2)
    bg_sample = _deterministic_sample([seg["key"] for seg in bg_segments_sorted], max_background_segments, seed_base + 3)

    by_key = {seg["key"]: seg for seg in all_segments_sorted}
    requested_keys = set(all_sample) | set(pos_sample) | set(bg_sample)
    requested_segments = [by_key[k] for k in requested_keys]

    # ── batch compute with optional parallelism ──
    seg_feature_map, n_resolved, n_total = acoustic_extractor.batch_compute_features(
        requested_segments,
        label=f"{aviary_name[:20]}|{species_norm}",
        aviary_hint=aviary_name,
    )
    missing = n_total - n_resolved

    local_resolved_fraction = float(n_resolved / n_total) if n_total > 0 else 0.0

    all_rows = [seg_feature_map[k] for k in all_sample if k in seg_feature_map]
    pos_rows = [seg_feature_map[k] for k in pos_sample if k in seg_feature_map]
    bg_rows = [seg_feature_map[k] for k in bg_sample if k in seg_feature_map]

    # choose the per-aviary target band before aggregation so the selected band
    # propagates into the main target-band contrasts and flock descriptors.
    adaptive = _compute_adaptive_band_features(
        pos_rows=pos_rows,
        bg_rows=bg_rows,
        species_norm=species_norm,
        wide_bands=acoustic_extractor.species_bands,
        narrow_bands=acoustic_extractor.narrow_bands,
    )
    selected_band = (
        float(adaptive.get("adaptive_band_fmin_hz", float("nan"))),
        float(adaptive.get("adaptive_band_fmax_hz", float("nan"))),
    )
    if not np.isfinite(selected_band[0]) or not np.isfinite(selected_band[1]):
        selected_band = acoustic_extractor.species_bands.get(
            species_norm, acoustic_extractor.species_bands.get("greater flamingo", (float("nan"), float("nan")))
        )

    agg_all = _aggregate_acoustic_rows(all_rows, selected_band)
    agg_pos = _aggregate_acoustic_rows(pos_rows, selected_band)
    agg_bg = _aggregate_acoustic_rows(bg_rows, selected_band)

    out = {
        "acoustic_all_window_count": len(all_rows),
        "acoustic_positive_window_count": len(pos_rows),
        "acoustic_background_window_count": len(bg_rows),
        "acoustic_windows_missing": missing,
        "acoustic_resolved_file_fraction": local_resolved_fraction,
        "target_band_fmin_hz": selected_band[0],
        "target_band_fmax_hz": selected_band[1],
    }

    for feat in MAAD_BASE_FEATURES:
        all_mean = agg_all.get(f"{feat}__mean", float("nan"))
        all_std = agg_all.get(f"{feat}__std", float("nan"))
        pos_mean = agg_pos.get(f"{feat}__mean", float("nan"))
        pos_median = agg_pos.get(f"{feat}__median", float("nan"))
        pos_p90 = agg_pos.get(f"{feat}__p90", float("nan"))
        bg_mean = agg_bg.get(f"{feat}__mean", float("nan"))
        bg_median = agg_bg.get(f"{feat}__median", float("nan"))
        out[f"{feat}__all_mean"] = all_mean
        out[f"{feat}__all_std"] = all_std
        out[f"{feat}__pos_mean"] = pos_mean
        out[f"{feat}__pos_median"] = pos_median
        out[f"{feat}__pos_p90"] = pos_p90
        out[f"{feat}__bg_mean"] = bg_mean
        out[f"{feat}__bg_median"] = bg_median
        out[f"{feat}__pos_minus_bg"] = float(pos_mean - bg_mean) if not (np.isnan(pos_mean) or np.isnan(bg_mean)) else float("nan")
        out[f"{feat}__pos_ratio_bg"] = _safe_ratio(pos_mean, bg_mean)
        out[f"{feat}__pos_minus_all"] = float(pos_mean - all_mean) if not (np.isnan(pos_mean) or np.isnan(all_mean)) else float("nan")
        out[f"{feat}__pos_ratio_all"] = _safe_ratio(pos_mean, all_mean)

    # ── composite acoustic features ──
    composite = _compute_composite_acoustic_features(out)
    out.update(composite)

    # ── flock-calling features ──
    flock = _compute_flock_features(
        pos_rows=pos_rows,
        bg_rows=bg_rows,
        selected_band=selected_band,
        cwr=cwr,
        occupancy=occupancy,
    )
    out.update(flock)

    # ── adaptive band selection (already used above, keep diagnostic columns) ──
    out.update(adaptive)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Composite acoustic features
# ═══════════════════════════════════════════════════════════════════════════

# Indices used in each composite.  Using ratio-form contrasts (pos_ratio_bg)
# as the basis since ratios are dimensionless and can be meaningfully averaged
# across indices with different native scales.

_SCENE_COMPLEXITY_INDICES = [
    "maad_aci",
    "maad_hf",
    "maad_nroi",
    "maad_evntcount",
]

_TARGET_CONTRAST_RATIO_INDICES = [
    "maad_aci",
    "maad_bi",
    "maad_nroi",
    "maad_hf",
    "maad_evnspcount",
]

_EVENT_DENSITY_INDICES = [
    "maad_evntcount",
    "maad_evnspcount",
]


def _finite_vals(values: List[float]) -> List[float]:
    return [v for v in values if v is not None and np.isfinite(v)]


def _safe_log1p_metric(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(np.log1p(max(value, 0.0)))


def _safe_log_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    if num is None or den is None or not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    return float(np.log((max(num, 0.0) + eps) / (max(den, 0.0) + eps)))


def _geometric_ratio_from_logs(values: List[float]) -> float:
    vals = _finite_vals(values)
    if not vals:
        return float("nan")
    return float(np.exp(np.mean(vals)))


def _compute_composite_acoustic_features(raw_block: Dict[str, float]) -> Dict[str, float]:
    """Derive five transformed composite scores from the raw acoustic block.

    The raw maad indices live on very different numeric scales (for example
    nROI versus Hf), so composites are built from log-compressed means and
    log-ratios rather than direct arithmetic averages of the original units.
    """
    scene_vals = _finite_vals([
        _safe_log1p_metric(raw_block.get(f"{idx}__all_mean", float("nan")))
        for idx in _SCENE_COMPLEXITY_INDICES
    ])
    scene_complexity = float(np.mean(scene_vals)) if scene_vals else float("nan")

    intensity_vals = _finite_vals([
        _safe_log1p_metric(raw_block.get(f"{idx}__pos_mean", float("nan")))
        for idx in _SCENE_COMPLEXITY_INDICES
    ])
    target_intensity = float(np.mean(intensity_vals)) if intensity_vals else float("nan")

    contrast_logs = [
        _safe_log_ratio(
            raw_block.get(f"{idx}__pos_mean", float("nan")),
            raw_block.get(f"{idx}__bg_mean", float("nan")),
        )
        for idx in _TARGET_CONTRAST_RATIO_INDICES
    ]
    target_contrast_ratio = _geometric_ratio_from_logs(contrast_logs)

    target_band_contrast = _geometric_ratio_from_logs([
        _safe_log_ratio(
            raw_block.get("maad_target_band_power_frac__pos_mean", float("nan")),
            raw_block.get("maad_target_band_power_frac__bg_mean", float("nan")),
        )
    ])

    event_logs = [
        _safe_log_ratio(
            raw_block.get(f"{idx}__pos_mean", float("nan")),
            raw_block.get(f"{idx}__bg_mean", float("nan")),
        )
        for idx in _EVENT_DENSITY_INDICES
    ]
    event_density_contrast = _geometric_ratio_from_logs(event_logs)

    return {
        "acoustic_scene_complexity": scene_complexity,
        "acoustic_target_intensity": target_intensity,
        "acoustic_target_contrast_ratio": target_contrast_ratio,
        "acoustic_target_band_contrast": target_band_contrast,
        "acoustic_event_density_contrast": event_density_contrast,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Flock-calling features
# ═══════════════════════════════════════════════════════════════════════════

def _compute_flock_features(
    pos_rows: List[Dict[str, float]],
    bg_rows: List[Dict[str, float]],
    selected_band: Tuple[float, float],
    cwr: float,
    occupancy: float,
) -> Dict[str, float]:
    """Compute flock-calling indicators from per-segment acoustic features."""
    band_key = _band_key_from_band(selected_band)

    def _extract(rows, key):
        return [float(r[key]) for r in rows if key in r and np.isfinite(r.get(key, float("nan")))]

    # 1. Energy stability from LEQt, converted from dB to linear power before CV.
    #    This keeps the variability measure on a ratio scale.
    leqt_pos_db = _extract(pos_rows, "maad_leqt")
    if len(leqt_pos_db) >= 5:
        leqt_pos_lin = np.power(10.0, np.asarray(leqt_pos_db, dtype=np.float64) / 10.0)
        cv = float(np.std(leqt_pos_lin)) / max(float(np.mean(leqt_pos_lin)), 1e-12)
        energy_stability = 1.0 / max(cv, 1e-6)
    else:
        energy_stability = float("nan")

    # 2. Event suppression: nROI ratio (positive / background).
    nroi_pos = _extract(pos_rows, "maad_nroi")
    nroi_bg = _extract(bg_rows, "maad_nroi")
    if nroi_pos and nroi_bg and float(np.mean(nroi_bg)) > 1e-6:
        event_suppression = float(np.mean(nroi_pos)) / float(np.mean(nroi_bg))
    else:
        event_suppression = float("nan")

    # 3. Spectral persistence: 1/CV of spectral activity fraction.
    actsp_pos = _extract(pos_rows, "maad_actspfract")
    if len(actsp_pos) >= 5:
        cv_sp = float(np.std(actsp_pos)) / max(float(np.mean(actsp_pos)), 1e-6)
        spectral_persistence = 1.0 / max(cv_sp, 1e-6)
    else:
        spectral_persistence = float("nan")

    # 4. Background bleed measured in the selected target band.
    band_act_bg = _extract(bg_rows, f"{band_key}__activity_frac")
    bg_bleed = float(np.mean(band_act_bg)) if band_act_bg else float("nan")

    # 5. Selected-band energy stability.
    band_pow_pos = _extract(pos_rows, f"{band_key}__power_frac")
    if len(band_pow_pos) >= 5:
        cv_bp = float(np.std(band_pow_pos)) / max(float(np.mean(band_pow_pos)), 1e-6)
        band_energy_stability = 1.0 / max(cv_bp, 1e-6)
    else:
        band_energy_stability = float("nan")

    components = []
    if np.isfinite(energy_stability):
        components.append(min(energy_stability, 50.0))
    if np.isfinite(event_suppression):
        inv = 1.0 / max(event_suppression, 0.1)
        components.append(min(inv, 10.0))
    if np.isfinite(spectral_persistence):
        components.append(min(spectral_persistence, 50.0))
    if np.isfinite(bg_bleed) and bg_bleed > 0:
        components.append(1.0 + bg_bleed * 10.0)
    if np.isfinite(band_energy_stability):
        components.append(min(band_energy_stability, 50.0))

    if components:
        flock_index = float(np.exp(np.mean(np.log(np.array(components, dtype=np.float64)))))
    else:
        flock_index = float("nan")

    if np.isfinite(flock_index) and np.isfinite(occupancy) and np.isfinite(cwr):
        correction = 1.0 + flock_index * (occupancy ** 2)
        flock_corrected_cwr = cwr * correction
    else:
        flock_corrected_cwr = float("nan")

    return {
        "flock_energy_stability": energy_stability,
        "flock_event_suppression": event_suppression,
        "flock_spectral_persistence": spectral_persistence,
        "flock_bg_bleed": bg_bleed,
        "flock_band_energy_stability": band_energy_stability,
        "flock_index": flock_index,
        "flock_corrected_cwr": flock_corrected_cwr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive band selection
# ═══════════════════════════════════════════════════════════════════════════

def _compute_adaptive_band_features(
    pos_rows: List[Dict[str, float]],
    bg_rows: List[Dict[str, float]],
    species_norm: str,
    wide_bands: Dict[str, Tuple[float, float]],
    narrow_bands: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Compare narrow and wide bands and select the one with more stable separation."""
    wide_band = wide_bands.get(species_norm, wide_bands.get("greater flamingo", (0.0, 0.0)))
    narrow_band = narrow_bands.get(species_norm)

    if narrow_band is None or narrow_band == wide_band:
        selected_band = wide_band
        contrasts = _band_contrast_metrics(pos_rows, bg_rows, selected_band)
        return {
            "adaptive_band_fmin_hz": selected_band[0],
            "adaptive_band_fmax_hz": selected_band[1],
            "adaptive_band_source": 1.0,
            "adaptive_band_power_frac_contrast": contrasts.get("power_frac", float("nan")),
            "adaptive_band_cover_contrast": contrasts.get("cover", float("nan")),
            "adaptive_band_activity_contrast": contrasts.get("activity_frac", float("nan")),
        }

    def _band_separation_score(rows_pos, rows_bg, band, metric="power_frac"):
        bk = _band_key_from_band(band)
        vals_pos = [float(r.get(f"{bk}__{metric}", float("nan"))) for r in rows_pos]
        vals_bg = [float(r.get(f"{bk}__{metric}", float("nan"))) for r in rows_bg]
        vals_pos = [v for v in vals_pos if np.isfinite(v)]
        vals_bg = [v for v in vals_bg if np.isfinite(v)]
        if not vals_pos or not vals_bg:
            return float("nan")
        mean_diff = float(np.mean(vals_pos) - np.mean(vals_bg))
        spread = float(np.std(vals_pos) + np.std(vals_bg))
        return mean_diff / max(spread, 1e-8)

    wide_score = _band_separation_score(pos_rows, bg_rows, wide_band, metric="power_frac")
    narrow_score = _band_separation_score(pos_rows, bg_rows, narrow_band, metric="power_frac")

    use_narrow = True
    if np.isfinite(wide_score) and np.isfinite(narrow_score):
        use_narrow = narrow_score >= wide_score
    elif np.isfinite(wide_score) and wide_score > 0:
        use_narrow = False

    selected_band = narrow_band if use_narrow else wide_band
    selected_source = 0.0 if use_narrow else 1.0
    contrasts = _band_contrast_metrics(pos_rows, bg_rows, selected_band)

    return {
        "adaptive_band_fmin_hz": selected_band[0],
        "adaptive_band_fmax_hz": selected_band[1],
        "adaptive_band_source": selected_source,
        "adaptive_band_power_frac_contrast": contrasts.get("power_frac", float("nan")),
        "adaptive_band_cover_contrast": contrasts.get("cover", float("nan")),
        "adaptive_band_activity_contrast": contrasts.get("activity_frac", float("nan")),
    }


def _band_contrast_metrics(
    rows_pos: List[Dict[str, float]],
    rows_bg: List[Dict[str, float]],
    band: Tuple[float, float],
) -> Dict[str, float]:
    """Return raw positive-minus-background contrasts for a band."""
    bk = _band_key_from_band(band)
    out = {}
    for metric in ["power_frac", "cover", "activity_frac"]:
        vals_pos = [float(r.get(f"{bk}__{metric}", float("nan"))) for r in rows_pos]
        vals_bg = [float(r.get(f"{bk}__{metric}", float("nan"))) for r in rows_bg]
        vals_pos = [v for v in vals_pos if np.isfinite(v)]
        vals_bg = [v for v in vals_bg if np.isfinite(v)]
        if vals_pos and vals_bg:
            out[metric] = float(np.mean(vals_pos) - np.mean(vals_bg))
        else:
            out[metric] = float("nan")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_target_species_features(
    aviary_name: str,
    ground_truth: AviaryGroundTruth,
    segments: List[Dict],
    detections: List[Dict],
    source_csv_keys: List[str],
    embedding_cache: EmbeddingCache,
    acoustic_extractor: Optional[AcousticIndexExtractor],
    bout_gap_seconds: float,
    max_acoustic_all_segments: int,
    max_acoustic_positive_segments: int,
    max_acoustic_background_segments: int,
) -> List[TargetSpeciesFeatures]:
    segment_map = {seg["key"]: seg for seg in segments}
    seg_det_idx = make_segment_detection_index(detections)
    species_detections = defaultdict(list)
    for det in detections:
        species_detections[det["species_normalized"]].append(det)

    audio_hours = compute_audio_hours(segments)
    coverage_hours = compute_coverage_hours(segments)
    total_files = len({seg["file"] for seg in segments})
    total_unique_hours = count_total_unique_hours(segments)
    total_audio_seconds = max(sum(seg["duration"] for seg in segments), 1e-8)

    features = []
    for species_norm, pop in ground_truth.species.items():
        if species_norm not in TARGET_SPECIES:
            continue

        matched_dets = list(species_detections.get(species_norm, []))
        if not matched_dets and species_norm in SPECIES_ALIASES:
            matched_dets = list(species_detections.get(SPECIES_ALIASES[species_norm], []))
        if not matched_dets:
            continue

        positive_segments = build_species_positive_segments(matched_dets, segment_map)
        if not positive_segments:
            continue
        bouts = build_bouts(positive_segments, gap_seconds=bout_gap_seconds)

        confs = np.array([d["confidence"] for d in matched_dets], dtype=np.float64)
        positive_segment_seconds = float(sum(seg["duration"] for seg in positive_segments))
        positive_audio_fraction = positive_segment_seconds / total_audio_seconds
        bout_span_fraction = float(sum(b["duration_sec"] for b in bouts)) / total_audio_seconds if bouts else 0.0

        longest_run_segments = int(max((b["n_segments"] for b in bouts), default=0))
        median_seg_dur = float(np.median([seg["duration"] for seg in positive_segments])) if positive_segments else 0.0
        longest_run_seconds = longest_run_segments * median_seg_dur
        inter_bout_gaps = []
        for a, b in zip(bouts[:-1], bouts[1:]):
            if a["end"] is not None and b["start"] is not None:
                inter_bout_gaps.append(max((b["start"] - a["end"]).total_seconds(), 0.0))

        ts_list = [seg["timestamp"] for seg in positive_segments if seg["timestamp"] is not None]
        active_hours = 0
        temporal_spread = 0.0
        peak_hour_share = 0.0
        if ts_list:
            hour_counts = defaultdict(int)
            for ts in ts_list:
                hour_counts[ts.replace(minute=0, second=0, microsecond=0)] += 1
            active_hours = len(hour_counts)
            temporal_spread = active_hours / max(total_unique_hours, 1)
            peak_hour_share = max(hour_counts.values()) / max(len(positive_segments), 1)

        labels_per_seg = []
        other_counts = []
        other_highconf_counts = []
        overlap_any = 0
        overlap_high = 0
        conf_margins = []
        positive_segment_keys = {seg["key"] for seg in positive_segments}
        for seg_key in positive_segment_keys:
            dets_here = seg_det_idx.get(seg_key, [])
            species_groups = defaultdict(list)
            for d in dets_here:
                species_groups[d["species_normalized"]].append(d["confidence"])
            label_count = len(species_groups)
            labels_per_seg.append(label_count)
            other_species = [sp for sp in species_groups if sp != species_norm]
            other_counts.append(len(other_species))
            high_other = [sp for sp in other_species if max(species_groups[sp]) >= HIGH_CONF_THRESHOLD]
            other_highconf_counts.append(len(high_other))
            if other_species:
                overlap_any += 1
            if high_other:
                overlap_high += 1
            target_best = max(species_groups.get(species_norm, [0.0]))
            other_best = max([max(species_groups[sp]) for sp in other_species], default=0.0)
            conf_margins.append(target_best - other_best)

        emb_vectors = []
        for csv_key in source_csv_keys:
            emb_index = embedding_cache.get_index(csv_key)
            if not emb_index:
                continue
            for seg in positive_segments:
                vec = emb_index.get((seg["file_base"], str(seg["segment"])))
                if vec is not None:
                    emb_vectors.append(vec)
        emb_feats = _embedding_feature_summary(emb_vectors)

        acoustic_payload = _build_acoustic_feature_block(
            species_norm=species_norm,
            aviary_name=aviary_name,
            segments=segments,
            positive_segments=positive_segments,
            acoustic_extractor=acoustic_extractor,
            max_all_segments=max_acoustic_all_segments,
            max_positive_segments=max_acoustic_positive_segments,
            max_background_segments=max_acoustic_background_segments,
            cwr=float(np.sum(confs)) / audio_hours,
            occupancy=positive_audio_fraction,
        )

        payload = {
            "aviary": aviary_name,
            "species": ground_truth.original_names.get(species_norm, species_norm),
            "scientific_name": ground_truth.scientific_names.get(species_norm, ""),
            "ground_truth_count": pop,
            "source_csv_keys": ";".join(sorted(set(source_csv_keys))),
            "recording_hours": audio_hours,
            "coverage_hours": coverage_hours,
            "n_total_segments": len(segments),
            "total_files": total_files,
            "total_detections": len(matched_dets),
            "detection_rate_per_hour": len(matched_dets) / audio_hours,
            "confidence_weighted_rate": float(np.sum(confs)) / audio_hours,
            "mean_confidence": float(np.mean(confs)),
            "std_confidence": float(np.std(confs)),
            "bout_count": len(bouts),
            "bout_rate_per_hour": len(bouts) / audio_hours,
            "total_bout_duration_sec": float(sum(b["duration_sec"] for b in bouts)),
            "mean_bout_duration_sec": _safe_mean([b["duration_sec"] for b in bouts]),
            "max_bout_duration_sec": float(max((b["duration_sec"] for b in bouts), default=0.0)),
            "mean_segments_per_bout": _safe_mean([b["n_segments"] for b in bouts]),
            "max_segments_per_bout": int(max((b["n_segments"] for b in bouts), default=0)),
            "active_files": len({seg["file"] for seg in positive_segments}),
            "active_file_fraction": len({seg["file"] for seg in positive_segments}) / max(total_files, 1),
            "active_hours": active_hours,
            "temporal_spread": temporal_spread,
            "positive_segment_seconds": positive_segment_seconds,
            "positive_audio_fraction": positive_audio_fraction,
            "bout_span_fraction": bout_span_fraction,
            "longest_positive_run_segments": longest_run_segments,
            "longest_positive_run_seconds": longest_run_seconds,
            "peak_hour_share_of_positive_segments": peak_hour_share,
            "mean_inter_bout_gap_sec": _safe_mean(inter_bout_gaps),
            "median_inter_bout_gap_sec": _safe_median(inter_bout_gaps),
            "min_inter_bout_gap_sec": float(min(inter_bout_gaps)) if inter_bout_gaps else 0.0,
            "mean_species_labels_per_positive_segment": _safe_mean(labels_per_seg, 1.0),
            "max_species_labels_per_positive_segment": int(max(labels_per_seg) if labels_per_seg else 1),
            "fraction_positive_segments_with_2plus_labels": float(np.mean(np.array(labels_per_seg) >= 2)) if labels_per_seg else 0.0,
            "fraction_positive_segments_with_3plus_labels": float(np.mean(np.array(labels_per_seg) >= 3)) if labels_per_seg else 0.0,
            "mean_other_species_per_positive_segment": _safe_mean(other_counts),
            "mean_highconf_other_species_per_positive_segment": _safe_mean(other_highconf_counts),
            "overlap_segments_any": overlap_any,
            "overlap_fraction_any": overlap_any / max(len(positive_segments), 1),
            "overlap_segments_highconf": overlap_high,
            "overlap_fraction_highconf": overlap_high / max(len(positive_segments), 1),
            "mean_target_conf_margin_vs_best_other": _safe_mean(conf_margins),
            **emb_feats,
            **acoustic_payload,
        }
        features.append(TargetSpeciesFeatures(payload=payload))
    return features


# ═══════════════════════════════════════════════════════════════════════════
# CSV output
# ═══════════════════════════════════════════════════════════════════════════

def save_features_csv(rows: List[TargetSpeciesFeatures], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "aviary",
            "species",
            "scientific_name",
            "ground_truth_count",
            "source_csv_keys",
        ] + FEATURE_COLUMNS
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row.payload["aviary"],
                row.payload["species"],
                row.payload["scientific_name"],
                row.payload["ground_truth_count"],
                row.payload["source_csv_keys"],
            ] + [row.payload.get(c, float("nan")) for c in FEATURE_COLUMNS])
    print(f"✓ stage-2 features saved to {out} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="BioDCASE 2026 — stage-2 feature builder (ARIA detection format)")
    parser.add_argument("--detections-dir", default=str(DEFAULT_PATHS["detections_dir"]))
    parser.add_argument("--aviary-csv", default=str(DEFAULT_PATHS["aviary_csv"]))
    parser.add_argument("--aviary-config", default=str(DEFAULT_PATHS["aviary_config"]),
                        help="Path to aviary_config.json with day mappings for BioDCASE filename resolution")
    parser.add_argument("--output", default=str(DEFAULT_PATHS["output_dir"] / "stage2_features.csv"))
    parser.add_argument("--bout-gap-seconds", type=float, default=DEFAULT_BOUT_GAP_SECONDS)
    parser.add_argument("--embeddings-dir", default=None, help="optional directory with NPZ embedding exports")
    parser.add_argument("--audio-root", action="append", default=[], help="root directory containing the original wav files. repeatable.")
    parser.add_argument("--max-acoustic-all-segments", type=int, default=1500)
    parser.add_argument("--max-acoustic-positive-segments", type=int, default=1500)
    parser.add_argument("--max-acoustic-background-segments", type=int, default=1500)
    parser.add_argument("--maad-max-sr", type=int, default=32000)
    parser.add_argument("--maad-nperseg", type=int, default=1024)
    parser.add_argument("--maad-noverlap", type=int, default=512)
    parser.add_argument("--maad-min-segment-duration", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help=(
            "Device for spectrogram computation.  'cuda' offloads torch.stft "
            "to GPU (requires torch+CUDA).  scikit-maad feature functions "
            "always run on CPU regardless of this setting."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of threads for parallel acoustic-index extraction.  "
            "Each thread reads audio, computes spectrograms, and runs maad "
            "features independently.  Recommended: 4-8 for I/O-bound setups."
        ),
    )
    args = parser.parse_args()

    aviaries = parse_aviary_csv(args.aviary_csv)
    aviary_day_mappings = load_aviary_day_mappings(args.aviary_config)

    # ── expand session-specific aliases (if any) ──
    for alias, base in AVIARY_GT_ALIAS.items():
        if base in aviaries and alias not in aviaries:
            gt = aviaries[base]
            aviaries[alias] = AviaryGroundTruth(
                aviary_name=alias,
                species=dict(gt.species),
                original_names=dict(gt.original_names),
                scientific_names=dict(gt.scientific_names),
            )
    embedding_cache = EmbeddingCache(args.embeddings_dir)

    acoustic_extractor = None
    if args.audio_root:
        # resolve device
        device = args.device
        if device == "cuda" and not TORCH_AVAILABLE:
            print("  --device cuda requested but torch+CUDA not available, falling back to CPU")
            device = "cpu"
        acoustic_extractor = AcousticIndexExtractor(
            audio_roots=args.audio_root,
            species_bands=DEFAULT_SPECIES_BANDS,
            narrow_bands=NARROW_SPECIES_BANDS,
            max_sample_rate=args.maad_max_sr,
            spectrogram_nperseg=args.maad_nperseg,
            spectrogram_noverlap=args.maad_noverlap,
            min_segment_duration=args.maad_min_segment_duration,
            device=device,
            max_workers=args.workers,
        )

    det_dir = Path(args.detections_dir)
    csv_files = sorted(det_dir.glob("*_detections.csv"))
    if not csv_files:
        csv_files = sorted(det_dir.glob("*.csv"))

    aviary_segments = defaultdict(list)
    aviary_detections = defaultdict(list)
    aviary_source_keys = defaultdict(list)

    print("=" * 88)
    print("BIODCASE 2026 — STAGE-2 FEATURE BUILDER (ARIA FORMAT)")
    print("=" * 88)
    print(f"detections dir: {det_dir}")
    print(f"aviary config: {args.aviary_config} ({len(aviary_day_mappings)} aviaries with day mappings)")
    print(f"csv files: {len(csv_files)}")
    if args.embeddings_dir:
        print(f"embeddings dir: {args.embeddings_dir}")
    if args.audio_root:
        print(f"audio roots: {args.audio_root}")
        print(f"scikit-maad available: {MAAD_AVAILABLE}")
        if acoustic_extractor:
            print(f"spectrogram device: {acoustic_extractor.device_name} (GPU={'active' if acoustic_extractor.use_gpu else 'off'})")
            print(f"parallel workers: {acoustic_extractor.max_workers}")
    print("=" * 88)
    print()

    for csv_file in csv_files:
        name = csv_file.stem
        # Strip standard suffixes to extract the aviary key
        for suffix in ["_detections", "_ensemble_voting_NEW", "_ensemble_voting", "_NEW"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if name not in CSV_TO_AVIARY:
            continue
        # Resolve per-aviary day mapping for BioDCASE filename format
        day_mapping = aviary_day_mappings.get(name, {})
        segments, detections = parse_aria_inference_csv(str(csv_file), day_mapping=day_mapping)
        aviary_names = CSV_TO_AVIARY[name]
        print(f"📄 {csv_file.name}")
        print(f"   → {aviary_names}")
        print(f"   {len(segments):,d} segments | {len(detections):,d} detections | {compute_audio_hours(segments):.2f} audio-h")
        for aviary_name in aviary_names:
            aviary_segments[aviary_name].extend(segments)
            aviary_detections[aviary_name].extend(detections)
            aviary_source_keys[aviary_name].append(name)

    rows = []
    print()
    print("=" * 88)
    print("EXTRACTING TARGET-SPECIES FEATURES")
    print("=" * 88)
    print()
    aviary_iterable = [name for name in sorted(aviary_segments.keys()) if name in aviaries]
    if tqdm is not None:
        aviary_iterable = tqdm(
            aviary_iterable,
            desc="target-species feature extraction",
            unit="aviary",
            dynamic_ncols=True,
        )

    for aviary_name in aviary_iterable:
        if tqdm is not None:
            aviary_iterable.set_postfix_str(aviary_name, refresh=False)
        seg_map = {seg["key"]: seg for seg in aviary_segments[aviary_name]}
        segments = list(seg_map.values())
        detections = aviary_detections[aviary_name]
        gt = aviaries[aviary_name]
        feats = extract_target_species_features(
            aviary_name=aviary_name,
            ground_truth=gt,
            segments=segments,
            detections=detections,
            source_csv_keys=aviary_source_keys[aviary_name],
            embedding_cache=embedding_cache,
            acoustic_extractor=acoustic_extractor,
            bout_gap_seconds=args.bout_gap_seconds,
            max_acoustic_all_segments=args.max_acoustic_all_segments,
            max_acoustic_positive_segments=args.max_acoustic_positive_segments,
            max_acoustic_background_segments=args.max_acoustic_background_segments,
        )
        if not feats:
            continue
        print(f"{aviary_name}: {len(feats)} target rows")
        for f in feats:
            p = f.payload
            print(
                f"  ✓ {p['species']:22s} pop={int(p['ground_truth_count']):4d} det/h={p['detection_rate_per_hour']:7.1f} "
                f"bout/h={p['bout_rate_per_hour']:7.1f} occ={100*p['positive_audio_fraction']:5.1f}% "
                f"flock={p.get('flock_index', float('nan')):.2f} "
                f"fCWR={p.get('flock_corrected_cwr', float('nan')):.0f} "
                f"band={'N' if p.get('adaptive_band_source', 1.0) == 0.0 else 'W'}"
            )
        rows.extend(feats)
        print()

    save_features_csv(rows, args.output)
    n_composites = len(COMPOSITE_ACOUSTIC_COLUMNS)
    n_raw_acoustic = len(MAAD_AGG_COLUMNS)
    print(f"\n stage-2 features saved to: {args.output}")
    print(f"  {len(rows)} rows | {len(FEATURE_COLUMNS)} features ({n_raw_acoustic} raw acoustic + {n_composites} composite)")
    if acoustic_extractor and acoustic_extractor.use_gpu:
        print(f"  spectrogram device: {acoustic_extractor.device_name}")
    print(f"\n  Run the estimator with:")
    print(f"  python estimator.py --features {args.output}")
    print()


if __name__ == "__main__":
    main()
