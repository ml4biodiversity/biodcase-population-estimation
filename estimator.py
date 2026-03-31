#!/usr/bin/env python3
"""
BioDCASE 2026 — Population Estimator (Stage 2)
================================================

Consumes the stage-2 target-species feature table and fits species-specific
population estimation models.  This is the final estimation step of the
BioDCASE 2026 Challenge population estimation baseline.

Five baseline models are provided, each being the best performer for at
least one (target species, detection format) combination:

+-----------------------------+----------------------+--------------------+
| Model                       | Best for (ARIA)      | Best for (BirdNET) |
+=============================+======================+====================+
| flock_corrected_cwr         | Greater flamingo     |                    |
| sim_weighted_cwr            |                      | Greater flamingo   |
| linear_coeff_bout_rate      | Hadada ibis          | Hadada ibis        |
| linear_coeff_cwr            | Red-billed quelea    |                    |
| adaptive_band_contrast      |                      | Red-billed quelea  |
+-----------------------------+----------------------+--------------------+

The best model depends on the upstream detector — participants should run
the full evaluation and consider which model suits their detection method.

Key design decisions
--------------------
- Three models use ``flock_corrected_cwr`` or ``adaptive_band_contrast``
  features, which require acoustic index extraction (``--audio-root`` in
  the feature builder).  If acoustic features are absent (NaN), these
  models are automatically skipped.

- All models use leave-one-out (LOO) cross-validation since the dataset
  contains only 6 aviaries.

- The estimator reports the best model per species based on lowest MAE,
  then a combined cross-species summary.

Usage
-----
::

    python estimator.py --features features/stage2_features.csv
"""

import csv
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

TARGET_SPECIES = {
    "greater flamingo",
    "red billed quelea",
    "hadada ibis",
}


STATIC_COLUMNS = {
    "aviary",
    "species",
    "scientific_name",
    "ground_truth_count",
    "source_csv_keys",
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
}

# -- Composite acoustic features --
# These are the 5 dimensionless summary scores computed by the feature builder.
COMPOSITE_ACOUSTIC_FEATURES = [
    "acoustic_scene_complexity",
    "acoustic_target_intensity",
    "acoustic_target_contrast_ratio",
    "acoustic_target_band_contrast",
    "acoustic_event_density_contrast",
]

# -- Flock-calling features --
FLOCK_FEATURES = [
    "flock_energy_stability",
    "flock_event_suppression",
    "flock_spectral_persistence",
    "flock_bg_bleed",
    "flock_band_energy_stability",
    "flock_index",
    "flock_corrected_cwr",
]

# -- Adaptive band features --
ADAPTIVE_BAND_FEATURES = [
    "adaptive_band_fmin_hz",
    "adaptive_band_fmax_hz",
    "adaptive_band_source",
    "adaptive_band_power_frac_contrast",
    "adaptive_band_cover_contrast",
    "adaptive_band_activity_contrast",
]

# Minimum fraction of points that must have non-NaN acoustic data for the
# acoustic-augmented model variants to run.
MIN_ACOUSTIC_COVERAGE = 0.5


@dataclass
class Stage2Point:
    aviary: str
    species: str
    species_display: str
    scientific_name: str
    pop: int
    source_csv_keys: str

    recording_hours: float
    coverage_hours: float
    n_total_segments: int
    total_files: int

    total_detections: int
    detection_rate_per_hour: float
    confidence_weighted_rate: float
    mean_confidence: float
    std_confidence: float

    bout_count: int
    bout_rate_per_hour: float
    total_bout_duration_sec: float
    mean_bout_duration_sec: float
    max_bout_duration_sec: float
    mean_segments_per_bout: float
    max_segments_per_bout: int

    active_files: int
    active_file_fraction: float
    active_hours: int
    temporal_spread: float

    positive_segment_seconds: float
    positive_audio_fraction: float
    bout_span_fraction: float

    longest_positive_run_segments: int
    longest_positive_run_seconds: float
    peak_hour_share_of_positive_segments: float

    mean_inter_bout_gap_sec: float
    median_inter_bout_gap_sec: float
    min_inter_bout_gap_sec: float

    mean_species_labels_per_positive_segment: float
    max_species_labels_per_positive_segment: int
    fraction_positive_segments_with_2plus_labels: float
    fraction_positive_segments_with_3plus_labels: float
    mean_other_species_per_positive_segment: float
    mean_highconf_other_species_per_positive_segment: float

    overlap_segments_any: int
    overlap_fraction_any: float
    overlap_segments_highconf: int
    overlap_fraction_highconf: float
    mean_target_conf_margin_vs_best_other: float

    extras: Dict[str, float] = field(default_factory=dict)

    def __getattr__(self, name: str):
        if name != "extras" and name in self.__dict__.get("extras", {}):
            return self.extras[name]
        raise AttributeError(name)


def normalize(name: str) -> str:
    return name.lower().replace("-", " ").replace("_", " ").strip()


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def load_stage2_features(path: str) -> List[Stage2Point]:
    points = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        extra_numeric_columns = [c for c in fieldnames if c not in STATIC_COLUMNS]
        for row in reader:
            sp_norm = normalize(row["species"])
            if sp_norm not in TARGET_SPECIES:
                continue
            extras = {c: _to_float(row, c, float("nan")) for c in extra_numeric_columns}
            points.append(
                Stage2Point(
                    aviary=row["aviary"],
                    species=sp_norm,
                    species_display=row["species"],
                    scientific_name=row.get("scientific_name", ""),
                    pop=int(_to_float(row, "ground_truth_count", 0)),
                    source_csv_keys=row.get("source_csv_keys", ""),
                    recording_hours=_to_float(row, "recording_hours", 0),
                    coverage_hours=_to_float(row, "coverage_hours", 0),
                    n_total_segments=int(_to_float(row, "n_total_segments", 0)),
                    total_files=int(_to_float(row, "total_files", 0)),
                    total_detections=int(_to_float(row, "total_detections", 0)),
                    detection_rate_per_hour=_to_float(row, "detection_rate_per_hour", 0),
                    confidence_weighted_rate=_to_float(row, "confidence_weighted_rate", 0),
                    mean_confidence=_to_float(row, "mean_confidence", 0),
                    std_confidence=_to_float(row, "std_confidence", 0),
                    bout_count=int(_to_float(row, "bout_count", 0)),
                    bout_rate_per_hour=_to_float(row, "bout_rate_per_hour", 0),
                    total_bout_duration_sec=_to_float(row, "total_bout_duration_sec", 0),
                    mean_bout_duration_sec=_to_float(row, "mean_bout_duration_sec", 0),
                    max_bout_duration_sec=_to_float(row, "max_bout_duration_sec", 0),
                    mean_segments_per_bout=_to_float(row, "mean_segments_per_bout", 0),
                    max_segments_per_bout=int(_to_float(row, "max_segments_per_bout", 0)),
                    active_files=int(_to_float(row, "active_files", 0)),
                    active_file_fraction=_to_float(row, "active_file_fraction", 0),
                    active_hours=int(_to_float(row, "active_hours", 0)),
                    temporal_spread=_to_float(row, "temporal_spread", 0),
                    positive_segment_seconds=_to_float(row, "positive_segment_seconds", 0),
                    positive_audio_fraction=_to_float(row, "positive_audio_fraction", 0),
                    bout_span_fraction=_to_float(row, "bout_span_fraction", 0),
                    longest_positive_run_segments=int(_to_float(row, "longest_positive_run_segments", 0)),
                    longest_positive_run_seconds=_to_float(row, "longest_positive_run_seconds", 0),
                    peak_hour_share_of_positive_segments=_to_float(row, "peak_hour_share_of_positive_segments", 0),
                    mean_inter_bout_gap_sec=_to_float(row, "mean_inter_bout_gap_sec", 0),
                    median_inter_bout_gap_sec=_to_float(row, "median_inter_bout_gap_sec", 0),
                    min_inter_bout_gap_sec=_to_float(row, "min_inter_bout_gap_sec", 0),
                    mean_species_labels_per_positive_segment=_to_float(row, "mean_species_labels_per_positive_segment", 0),
                    max_species_labels_per_positive_segment=int(_to_float(row, "max_species_labels_per_positive_segment", 0)),
                    fraction_positive_segments_with_2plus_labels=_to_float(row, "fraction_positive_segments_with_2plus_labels", 0),
                    fraction_positive_segments_with_3plus_labels=_to_float(row, "fraction_positive_segments_with_3plus_labels", 0),
                    mean_other_species_per_positive_segment=_to_float(row, "mean_other_species_per_positive_segment", 0),
                    mean_highconf_other_species_per_positive_segment=_to_float(row, "mean_highconf_other_species_per_positive_segment", 0),
                    overlap_segments_any=int(_to_float(row, "overlap_segments_any", 0)),
                    overlap_fraction_any=_to_float(row, "overlap_fraction_any", 0),
                    overlap_segments_highconf=int(_to_float(row, "overlap_segments_highconf", 0)),
                    overlap_fraction_highconf=_to_float(row, "overlap_fraction_highconf", 0),
                    mean_target_conf_margin_vs_best_other=_to_float(row, "mean_target_conf_margin_vs_best_other", 0),
                    extras=extras,
                )
            )
    return [p for p in points if p.total_detections > 0]


# ========================================================================
# Acoustic / flock coverage helpers
# ========================================================================

def _acoustic_coverage(points: List[Stage2Point]) -> Tuple[int, int, float]:
    """Return (n_with_acoustic, n_total, fraction) for a list of points."""
    n = len(points)
    if n == 0:
        return 0, 0, 0.0
    n_with = sum(
        1 for p in points
        if not np.isnan(getattr(p, "acoustic_scene_complexity", float("nan")))
    )
    return n_with, n, n_with / n


def _has_enough_acoustic(points: List[Stage2Point]) -> bool:
    n_with, n, frac = _acoustic_coverage(points)
    return frac >= MIN_ACOUSTIC_COVERAGE and n_with >= 2


def _flock_coverage(points: List[Stage2Point]) -> Tuple[int, int, float]:
    n = len(points)
    if n == 0:
        return 0, 0, 0.0
    n_with = sum(
        1 for p in points
        if not np.isnan(getattr(p, "flock_index", float("nan")))
    )
    return n_with, n, n_with / n


def _has_flock_features(points: List[Stage2Point]) -> bool:
    n_with, n, frac = _flock_coverage(points)
    return frac >= MIN_ACOUSTIC_COVERAGE and n_with >= 2


# ========================================================================
# Numerical helpers
# ========================================================================

def _nanmean_no_warn(X: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(X, axis=axis)


def _nanstd_no_warn(X: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanstd(X, axis=axis)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "MAPE": float("nan"), "N": 0}
    res = y_true - y_pred
    mae = float(np.mean(np.abs(res)))
    rmse = float(np.sqrt(np.mean(res ** 2)))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    nz = y_true > 0
    mape = float(np.mean(np.abs(res[nz] / y_true[nz])) * 100) if np.any(nz) else float("inf")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "N": int(len(y_true))}


def _has_usable_feature(points: List[Stage2Point], feature: str) -> bool:
    vals = [getattr(p, feature, float("nan")) for p in points]
    vals = [v for v in vals if not np.isnan(v)]
    return len(vals) >= 2 and np.nanstd(vals) > 1e-12


# ========================================================================
# Feature lists for the models
# ========================================================================

# Structure features used by similarity-weighted model
_STRUCTURE_FEATURES_BASE = [
    "positive_audio_fraction",
    "bout_span_fraction",
    "longest_positive_run_segments",
    "peak_hour_share_of_positive_segments",
    "mean_inter_bout_gap_sec",
    "mean_species_labels_per_positive_segment",
    "mean_highconf_other_species_per_positive_segment",
    "overlap_fraction_any",
    "overlap_fraction_highconf",
    "mean_target_conf_margin_vs_best_other",
]

_LOG_FEATURES_BASE = {
    "longest_positive_run_segments",
    "mean_inter_bout_gap_sec",
}

# Features that can take negative values (contrasts)
SIGNED_COEFFICIENT_FEATURES = {
    "adaptive_band_power_frac_contrast",
    "adaptive_band_cover_contrast",
    "adaptive_band_activity_contrast",
    "maad_nroi__pos_minus_bg",
}


# ========================================================================
# Models
# ========================================================================

class LinearCoefficientModel:
    """Simple per-individual rate model: feature_value / population = constant.

    For each LOO fold, computes the per-individual rate from the training
    points, then predicts the held-out point as feature_value / mean_rate.

    Used by: linear_coeff_cwr, linear_coeff_bout_rate, flock_corrected_cwr,
    adaptive_band_contrast.
    """

    def __init__(self, feature: str, allow_negative: Optional[bool] = None):
        self.feature = feature
        self.allow_negative = feature in SIGNED_COEFFICIENT_FEATURES if allow_negative is None else allow_negative
        tag = ", signed" if self.allow_negative else ""
        self.name = f"linear_coeff({feature}{tag})"

    def _get_feature(self, p: Stage2Point) -> float:
        return getattr(p, self.feature, float("nan"))

    def fit_and_predict_loo(self, points: List[Stage2Point]):
        n = len(points)
        y_true = np.array([p.pop for p in points], dtype=np.float64)
        y_pred = np.full(n, np.nan)
        per_indiv = []
        for p in points:
            v = self._get_feature(p)
            if p.pop <= 0 or np.isnan(v):
                per_indiv.append(float("nan"))
            elif self.allow_negative:
                per_indiv.append(v / p.pop)
            else:
                per_indiv.append(v / p.pop if v > 0 else float("nan"))

        valid = [v for v in per_indiv if np.isfinite(v) and (self.allow_negative or v > 0)]
        if valid:
            global_coeff = float(np.mean(valid))
            global_cv = float(np.std(valid) / max(abs(global_coeff), 1e-12))
        else:
            global_coeff = float("nan")
            global_cv = float("inf")

        for i, p in enumerate(points):
            other = [
                per_indiv[j] for j in range(n)
                if j != i and np.isfinite(per_indiv[j]) and (self.allow_negative or per_indiv[j] > 0)
            ]
            if not other:
                continue
            coeff = float(np.mean(other))
            feat_val = self._get_feature(p)
            if np.isnan(feat_val) or abs(coeff) <= 1e-12:
                continue
            pred = feat_val / coeff
            if np.isfinite(pred):
                y_pred[i] = max(round(float(pred)), 0)
        return y_true, y_pred, {
            "global_coeff": global_coeff,
            "global_cv": global_cv,
            "allow_negative": self.allow_negative,
        }


class SimilarityWeightedCoefficientModel:
    """Similarity-weighted per-individual rate model.

    Like LinearCoefficientModel but weights training-point contributions
    by structural similarity (inverse Euclidean distance in a normalised
    feature space).  Points with similar acoustic/temporal structure get
    higher influence on the predicted per-individual rate.

    Uses detection-only structure features (occupancy, bout patterns,
    species overlap, etc.) for the similarity computation.

    Used by: sim_weighted_cwr.
    """

    def __init__(self, feature: str = "confidence_weighted_rate"):
        self.feature = feature
        self.name = f"sim_weighted({feature})"
        self.structure_features = list(_STRUCTURE_FEATURES_BASE)
        self.log_features = set(_LOG_FEATURES_BASE)

    def _feature_value(self, p: Stage2Point) -> float:
        return getattr(p, self.feature, float("nan"))

    def _structure_matrix(self, points: List[Stage2Point]) -> np.ndarray:
        X = np.array([[getattr(p, f, float("nan")) for f in self.structure_features] for p in points], dtype=np.float64)
        for j, name in enumerate(self.structure_features):
            if name in self.log_features:
                X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
        return X

    def fit_and_predict_loo(self, points: List[Stage2Point]):
        n = len(points)
        if n < 2:
            return None, None, {"reason": "n<2"}
        y_true = np.array([p.pop for p in points], dtype=np.float64)
        y_pred = np.full(n, np.nan)

        X = self._structure_matrix(points)
        for i, p in enumerate(points):
            train_idx = [j for j in range(n) if j != i]
            Xtr = X[train_idx]
            mu = _nanmean_no_warn(Xtr, axis=0)
            mu = np.where(np.isnan(mu), 0.0, mu)
            sigma = _nanstd_no_warn(Xtr, axis=0)
            sigma = np.where(np.isnan(sigma), 1.0, sigma)
            sigma[sigma < 1e-8] = 1.0
            Xtr_imp = np.where(np.isnan(Xtr), mu, Xtr)
            xt = np.where(np.isnan(X[i]), mu, X[i])
            Xtr_z = (Xtr_imp - mu) / sigma
            xt_z = (xt - mu) / sigma

            coeffs = []
            dists = []
            for loc, j in enumerate(train_idx):
                feat_val = self._feature_value(points[j])
                if points[j].pop <= 0 or np.isnan(feat_val) or feat_val <= 0:
                    continue
                coeffs.append(feat_val / points[j].pop)
                dists.append(float(np.linalg.norm(xt_z - Xtr_z[loc])))
            if not coeffs:
                continue
            coeffs = np.asarray(coeffs)
            dists = np.asarray(dists)
            weights = 1.0 / np.maximum(dists, 1e-6)
            coeff = float(np.sum(weights * coeffs) / np.sum(weights))
            feat_val = self._feature_value(p)
            if coeff > 0 and not np.isnan(feat_val):
                y_pred[i] = max(round(feat_val / coeff), 0)

        return y_true, y_pred, {
            "structure_features": self.structure_features,
        }


# ========================================================================
# Reporting
# ========================================================================

def _print_predictions(points, y_true, y_pred, metrics):
    print(f"    {'Aviary':<32s} {'True':>5s} {'Pred':>8s} {'Err':>8s}")
    for i, p in enumerate(points):
        pred = y_pred[i]
        err = pred - y_true[i] if not np.isnan(pred) else float('nan')
        print(f"    {p.aviary[:30]:<32s} {y_true[i]:5.0f} {pred:8.0f} {err:+8.0f}")
    print(f"    MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.1f}%  N={metrics['N']}")


def analyze_species(species_name, points):
    n = len(points)
    n_ac, _, frac_ac = _acoustic_coverage(points)
    n_flock, _, frac_flock = _flock_coverage(points)
    has_flock = _has_flock_features(points)

    print()
    print("=" * 108)
    print(
        f"  {species_name.upper()} -- {n} aviary data points  |  acoustic: {n_ac}/{n} ({100*frac_ac:.0f}%)"
        f"  |  flock: {n_flock}/{n} ({100*frac_flock:.0f}%)"
    )
    print("=" * 108)
    print()

    # -- data overview --
    print(f"  {'Aviary':<32s} {'Pop':>5s} {'CWR/h':>8s} {'Bout/h':>8s} {'Occ%':>7s} {'FlockIdx':>9s} {'fCWR/h':>8s} {'Band':>5s} {'ABCon':>8s}")
    print(f"  {'-'*112}")
    for p in sorted(points, key=lambda x: x.pop, reverse=True):
        fi = getattr(p, "flock_index", float("nan"))
        fcwr = getattr(p, "flock_corrected_cwr", float("nan"))
        ab_src = getattr(p, "adaptive_band_source", float("nan"))
        ab_con = getattr(p, "adaptive_band_power_frac_contrast", float("nan"))
        band_label = "N" if ab_src == 0.0 else ("W" if ab_src == 1.0 else "?")
        print(
            f"  {p.aviary[:30]:<32s} {p.pop:5d} {p.confidence_weighted_rate:8.1f} {p.bout_rate_per_hour:8.1f} "
            f"{100*p.positive_audio_fraction:6.1f}% {fi:9.2f} {fcwr:8.0f} {band_label:>5s} {ab_con:8.4f}"
        )

    results = {}

    # ================================================================
    # Model 1: Linear CWR coefficient (detection-only)
    #
    # Assumes each individual contributes a constant confidence-weighted
    # detection rate (CWR) per hour.  Best for species with consistent
    # individual calling behaviour (Red-billed quelea with ARIA).
    # ================================================================

    if n >= 2:
        model = LinearCoefficientModel(feature="confidence_weighted_rate")
        yt, yp, info = model.fit_and_predict_loo(points)
        m = evaluate(yt, yp)
        results["linear_coeff_cwr"] = (points, yt, yp, m, info)
        print("\n  Model 1: Linear CWR coefficient (LOO)")
        print(f"    CWR / individual: {info['global_coeff']:.4f}  CV: {info['global_cv']:.3f}")
        _print_predictions(points, yt, yp, m)

    # ================================================================
    # Model 2: Linear bout-rate coefficient (detection-only)
    #
    # Assumes each individual contributes a constant number of calling
    # bouts per hour.  Best for species with distinct, countable calling
    # events (Hadada ibis with both ARIA and BirdNET).
    # ================================================================

    if n >= 2:
        model = LinearCoefficientModel(feature="bout_rate_per_hour")
        yt, yp, info = model.fit_and_predict_loo(points)
        m = evaluate(yt, yp)
        results["linear_coeff_bout_rate"] = (points, yt, yp, m, info)
        print("\n  Model 2: Linear bout-rate coefficient (LOO)")
        print(f"    bouts / individual / h: {info['global_coeff']:.4f}  CV: {info['global_cv']:.3f}")
        _print_predictions(points, yt, yp, m)

    # ================================================================
    # Model 3: Flock-corrected CWR (requires acoustic features)
    #
    # For flock-calling species (flamingos), many individuals call
    # synchronously within the same segment, suppressing per-individual
    # CWR.  The flock-corrected CWR inflates raw CWR proportional to
    # flock index x occupancy^2, compensating for this suppression.
    # Best for Greater flamingo with ARIA detections.
    # ================================================================

    if n >= 2 and has_flock:
        if _has_usable_feature(points, "flock_corrected_cwr"):
            model = LinearCoefficientModel(feature="flock_corrected_cwr")
            yt, yp, info = model.fit_and_predict_loo(points)
            m = evaluate(yt, yp)
            results["flock_corrected_cwr"] = (points, yt, yp, m, info)
            print(f"\n  Model 3: Flock-corrected CWR coefficient (LOO)")
            print(f"    flock CWR / individual: {info['global_coeff']:.4f}  CV: {info['global_cv']:.3f}")
            _print_predictions(points, yt, yp, m)
        else:
            print(f"\n  Model 3: SKIPPED -- flock_corrected_cwr has insufficient variance")
    else:
        if n >= 2:
            print(f"\n  Model 3: SKIPPED -- flock coverage {n_flock}/{n} below threshold")

    # ================================================================
    # Model 4: Similarity-weighted CWR (detection-only)
    #
    # Weights the per-individual CWR rate by structural similarity
    # (occupancy, bout patterns, species overlap) between aviaries.
    # Best for Greater flamingo with BirdNET detections.
    # ================================================================

    if n >= 3:
        model = SimilarityWeightedCoefficientModel(feature="confidence_weighted_rate")
        yt, yp, info = model.fit_and_predict_loo(points)
        m = evaluate(yt, yp)
        results["sim_weighted_cwr"] = (points, yt, yp, m, info)
        print("\n  Model 4: Similarity-weighted CWR (detection-only, LOO)")
        _print_predictions(points, yt, yp, m)
    else:
        print(f"\n  Model 4: SKIPPED -- requires n>=3 data points (have {n})")

    # ================================================================
    # Model 5: Adaptive-band contrast (requires acoustic features)
    #
    # Uses the positive-minus-background power fraction contrast in the
    # adaptively-selected frequency band as a direct population proxy.
    # Best for Red-billed quelea with BirdNET detections.
    # ================================================================

    if n >= 2 and has_flock:
        if _has_usable_feature(points, "adaptive_band_power_frac_contrast"):
            model = LinearCoefficientModel(feature="adaptive_band_power_frac_contrast")
            yt, yp, info = model.fit_and_predict_loo(points)
            if yt is not None and not np.all(np.isnan(yp)):
                m = evaluate(yt, yp)
                results["adaptive_band_contrast"] = (points, yt, yp, m, info)
                print(f"\n  Model 5: Adaptive-band contrast coefficient (LOO)")
                print(f"    band contrast / individual: {info['global_coeff']:.6f}  CV: {info['global_cv']:.3f}")
                _print_predictions(points, yt, yp, m)
        else:
            print(f"\n  Model 5: SKIPPED -- adaptive_band_power_frac_contrast has insufficient variance")
    else:
        if n >= 2:
            print(f"\n  Model 5: SKIPPED -- acoustic coverage below threshold")

    # ================================================================
    # Summary table
    # ================================================================

    if results:
        print(f"\n  {'-'*92}")
        print(f"  MODEL COMPARISON for {species_name}   [acoustic: {n_ac}/{n}]")
        print(f"  {'-'*92}")
        print(f"  {'Model':<42s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'MAPE':>8s}")
        print(f"  {'-'*88}")

        for name, (_, _, _, m, _) in results.items():
            print(f"  {name:<42s} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['R2']:8.4f} {m['MAPE']:7.1f}%")
    return results


# ========================================================================
# Main
# ========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BioDCASE 2026 -- population estimator (stage 2)")
    parser.add_argument("--features", default="features/stage2_features.csv")
    args = parser.parse_args()

    print("=" * 108)
    print("BIODCASE 2026 -- POPULATION ESTIMATOR (STAGE 2)")
    print("=" * 108)

    all_points = load_stage2_features(args.features)
    print(f"\nloaded {len(all_points)} relevant detected rows from {args.features}")

    # -- global acoustic coverage report --
    n_ac_global, n_global, frac_global = _acoustic_coverage(all_points)
    print(f"acoustic coverage: {n_ac_global}/{n_global} rows ({100*frac_global:.0f}%) have non-NaN composite features")
    if frac_global < MIN_ACOUSTIC_COVERAGE:
        print(f"  warning: below {MIN_ACOUSTIC_COVERAGE:.0%} threshold -- acoustic-augmented models (3, 5) will be skipped")
        print(f"     -> pass --audio-root to the feature builder to enable acoustic features")
    print()

    by_species = defaultdict(list)
    for p in all_points:
        by_species[p.species].append(p)

    print(f"  {'Species':<25s} {'#Aviary':>7s} {'Pop range':>12s} {'Total det':>10s} {'Bouts':>10s} {'AcCov':>6s}")
    print(f"  {'-'*80}")
    for sp in sorted(by_species.keys()):
        pts = by_species[sp]
        pops = [p.pop for p in pts]
        n_ac_sp, _, _ = _acoustic_coverage(pts)
        print(
            f"  {sp:<25s} {len(pts):7d} {min(pops):5d}-{max(pops):5d}"
            f" {sum(p.total_detections for p in pts):10,d}"
            f" {sum(p.bout_count for p in pts):10,d}"
            f" {n_ac_sp:3d}/{len(pts)}"
        )


    all_results = {}
    for species in sorted(TARGET_SPECIES):
        pts = by_species.get(species, [])
        if not pts:
            print(f"\n  warning: {species}: no detected data points, skipping")
            continue
        all_results[species] = analyze_species(species, pts)

    # ================================================================
    # Final combined report
    # ================================================================

    print("\n\n" + "=" * 108)
    print("FINAL COMBINED REPORT -- BEST MODEL PER TARGET SPECIES")
    print("=" * 108)

    combined_true = []
    combined_pred = []

    print(f"\n  {'Species':<22s} {'Best model':<42s} {'Aviary':<28s} {'True':>5s} {'Pred':>7s} {'Err':>7s}")
    print(f"  {'-'*118}")
    for species in sorted(TARGET_SPECIES):
        res = all_results.get(species)
        if not res:
            continue
        valid = {k: v for k, v in res.items() if v[3]['N'] > 0 and not np.isnan(v[3]['MAE'])}
        if not valid:
            continue
        best_name = min(valid, key=lambda k: valid[k][3]['MAE'])
        pts, yt, yp, m, info = valid[best_name]
        for i, p in enumerate(pts):
            pred = yp[i]
            err = pred - yt[i] if not np.isnan(pred) else float('nan')
            combined_true.append(yt[i])
            combined_pred.append(pred)
            print(f"  {species[:20]:<22s} {best_name[:40]:<42s} {p.aviary[:26]:<28s} {yt[i]:5.0f} {pred:7.0f} {err:+7.0f}")

    if combined_true:
        ct = np.asarray(combined_true, dtype=np.float64)
        cp = np.asarray(combined_pred, dtype=np.float64)
        m_all = evaluate(ct, cp)
        print(f"\n  combined: MAE={m_all['MAE']:.2f}  RMSE={m_all['RMSE']:.2f}  R2={m_all['R2']:.4f}  MAPE={m_all['MAPE']:.1f}%  N={m_all['N']}")


if __name__ == '__main__':
    main()
