#!/usr/bin/env python3
"""
BioDCASE 2026 — Stage-2 Feature Builder (BirdNET detection format)
===================================================================

Runs the identical stage-2 population-estimation feature pipeline on
**default BirdNET** detection CSVs instead of ARIA ensemble-voting CSVs.

This script imports all shared logic (acoustic extraction, embedding helpers,
feature computation, etc.) from the ARIA feature builder and only overrides:

1. The CSV parser — ``parse_birdnet_csv()`` reads default BirdNET output
   format: ``Start (s), End (s), Scientific name, Common name, Confidence, File``
2. The CSV-to-aviary mapping — ``BIRDNET_CSV_TO_AVIARY``
3. Default paths

Detection CSV naming convention
--------------------------------
Participants must name their detection CSVs as::

    dev_aviary_1_detections.csv
    dev_aviary_2_detections.csv
    ...
    dev_aviary_6_detections.csv

BirdNET detection format (from ``pip install aria-inference-birdnet`` or
default BirdNET)::

    Start (s),End (s),Scientific name,Common name,Confidence,File

Usage
-----
::

    python feature_builder_birdnet.py \\
        --detections-dir ./birdnet_detections \\
        --output features/stage2_features_birdnet.csv \\
        --audio-root /path/to/audio \\
        --device cuda --workers 4

Then run the estimator on the BirdNET features::

    python estimator.py --features features/stage2_features_birdnet.csv

Notes
-----
- BirdNET CSVs only contain segments where at least one species exceeded the
  confidence threshold.  Segments with no detection are absent, so
  ``n_total_segments`` and ``recording_hours`` reflect *detected* segments
  only (not total analysed windows).  This is expected for a baseline
  comparison — the detection-derived features (CWR, bout rate, etc.) remain
  valid and comparable.

- BirdNET uses Common name directly (e.g. "Red-billed Quelea"); the
  ``normalize_species_name()`` function handles lowercasing and hyphen→space
  conversion identically to the ARIA pipeline.

- If your BirdNET CSV filenames differ from the mapping below, update
  ``BIRDNET_CSV_TO_AVIARY`` accordingly.
"""

import csv
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import everything from the ARIA feature builder ──
# This gives us: all data classes, utility functions, acoustic extractor,
# embedding cache, feature extraction logic, CSV output, etc.
from feature_builder import (
    # data classes
    AviaryGroundTruth,
    TargetSpeciesFeatures,
    # parsing / utility
    normalize_species_name,
    parse_aviary_csv,
    load_aviary_day_mappings,
    parse_filename_timestamp,
    compute_audio_hours,
    compute_coverage_hours,
    # embedding support
    EmbeddingCache,
    CSV_TO_EMBEDDING_PREFIX,
    # acoustic extractor
    AcousticIndexExtractor,
    MAAD_AVAILABLE,
    # feature extraction core
    extract_target_species_features,
    save_features_csv,
    # constants
    DEFAULT_SPECIES_BANDS,
    NARROW_SPECIES_BANDS,
    AVIARY_GT_ALIAS,
    FEATURE_COLUMNS,
    TARGET_SPECIES,
    ALL_RELEVANT,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# BirdNET-specific configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_BIRDNET_PATHS = {
    "aviary_csv": Path(__file__).parent / "ground_truth.csv",
    "aviary_config": Path(__file__).parent / "aviary_config.json",
    "detections_dir": Path(__file__).parent / "detections",
    "output_dir": Path(__file__).parent / "features",
    "embeddings_dir": None,
}

# ── BirdNET CSV filename → aviary name mapping ──
# Detection CSVs should be named ``dev_aviary_N_detections.csv``.
# The suffix is stripped automatically; see ``_strip_birdnet_suffix()``.
BIRDNET_CSV_TO_AVIARY: Dict[str, List[str]] = {
    "dev_aviary_1": ["dev_aviary_1"],
    "dev_aviary_2": ["dev_aviary_2"],
    "dev_aviary_3": ["dev_aviary_3"],
    "dev_aviary_4": ["dev_aviary_4"],
    "dev_aviary_5": ["dev_aviary_5"],
    "dev_aviary_6": ["dev_aviary_6"],
}

# ── BirdNET CSV → embedding prefix ──
# Embeddings were extracted from the same audio, so the prefix mapping
# matches the ARIA one.
BIRDNET_CSV_TO_EMBEDDING_PREFIX: Dict[str, str] = {
    "dev_aviary_1": "dev_aviary_1",
    "dev_aviary_2": "dev_aviary_2",
    "dev_aviary_3": "dev_aviary_3",
    "dev_aviary_4": "dev_aviary_4",
    "dev_aviary_5": "dev_aviary_5",
    "dev_aviary_6": "dev_aviary_6",
}


# ═══════════════════════════════════════════════════════════════════════════
# BirdNET CSV parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_birdnet_csv(csv_path: str, day_mapping: Optional[Dict[str, str]] = None) -> Tuple[List[Dict], List[Dict]]:
    """Parse a default BirdNET merged results CSV.

    BirdNET format::

        Start (s),End (s),Scientific name,Common name,Confidence,File

    Returns the same ``(segments, detections)`` tuple format as
    ``parse_aria_inference_csv()`` so the downstream pipeline is identical.

    Key differences from ARIA:
    - No "Status" column — every row is a valid detection.
    - No "Segment" column — we synthesise a segment ID from the
      (file, start) pair.
    - "File" column may contain full absolute paths.
    - Species comes from "Common name" (BirdNET naming convention).
    - Only segments with at least one detection appear in the CSV.
    """
    segment_map: Dict[Tuple, Dict] = {}
    detections: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Handle slight column-name variations across BirdNET versions
        # (e.g. "File" vs "File name", "Start (s)" vs "Start(s)")
        fieldnames = reader.fieldnames or []
        # Build a normalised lookup: strip whitespace + lowercase
        _norm_fields = {fn.strip().lower(): fn for fn in fieldnames}

        def _col(candidates: List[str]) -> Optional[str]:
            """Find the first matching column name."""
            for c in candidates:
                if c.strip().lower() in _norm_fields:
                    return _norm_fields[c.strip().lower()]
            return None

        col_start = _col(["Start (s)", "Start(s)", "Start"])
        col_end = _col(["End (s)", "End(s)", "End"])
        col_common = _col(["Common name", "Common Name", "Common"])
        col_conf = _col(["Confidence", "confidence"])
        col_file = _col(["File", "File name", "Filename", "filepath"])

        if not all([col_start, col_end, col_common, col_conf, col_file]):
            missing = []
            if not col_start: missing.append("Start (s)")
            if not col_end: missing.append("End (s)")
            if not col_common: missing.append("Common name")
            if not col_conf: missing.append("Confidence")
            if not col_file: missing.append("File")
            print(f"   BirdNET CSV {csv_path} missing columns: {missing}")
            print(f"      available columns: {fieldnames}")
            return [], []

        for row in reader:
            file_path = row.get(col_file, "").strip()
            if not file_path:
                continue

            try:
                start = float(row.get(col_start, 0))
                end = float(row.get(col_end, 0))
            except (ValueError, TypeError):
                continue

            duration = max(end - start, 0.0)
            file_base = Path(file_path).name

            # Extract timestamp from filename (same pattern as ARIA)
            timestamp = parse_filename_timestamp(file_base, day_mapping=day_mapping)
            abs_start = timestamp + timedelta(seconds=start) if timestamp else None
            abs_end = timestamp + timedelta(seconds=end) if timestamp else None

            # Synthesise segment ID — BirdNET doesn't provide one.
            # Use start time (in centiseconds) for uniqueness within a file.
            segment_id = str(int(round(start * 100)))
            seg_key = (file_path, segment_id, start, end)

            if seg_key not in segment_map:
                segment_map[seg_key] = {
                    "file": file_path,
                    "file_base": file_base,
                    "segment": segment_id,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "timestamp": timestamp,
                    "abs_start": abs_start,
                    "abs_end": abs_end,
                    "key": seg_key,
                }

            common_name = row.get(col_common, "").strip()
            if not common_name:
                continue

            try:
                confidence = float(row.get(col_conf, 0))
            except (ValueError, TypeError):
                continue

            detections.append({
                "file": file_path,
                "file_base": file_base,
                "segment": segment_id,
                "species_raw": common_name,
                "species_normalized": normalize_species_name(common_name),
                "confidence": confidence,
                "start": start,
                "end": end,
                "timestamp": timestamp,
                "abs_start": abs_start,
                "abs_end": abs_end,
                "segment_key": seg_key,
            })

    return list(segment_map.values()), detections


def _strip_birdnet_suffix(stem: str) -> str:
    """Strip known BirdNET suffixes to extract the aviary identifier.

    Handles patterns like:
    - ``dev_aviary_1_detections`` → ``dev_aviary_1``
    - ``dev_aviary_1_all_results`` → ``dev_aviary_1``
    - ``dev_aviary_1_filtered_results`` → ``dev_aviary_1``
    """
    for suffix in [
        "_detections",
        "_all_results",
        "_filtered_results",
        "_results",
        "_birdnet",
        "_BirdNET",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ═══════════════════════════════════════════════════════════════════════════
# Custom EmbeddingCache that uses BirdNET prefix mapping
# ═══════════════════════════════════════════════════════════════════════════

class BirdNETEmbeddingCache(EmbeddingCache):
    """Like EmbeddingCache but uses BIRDNET_CSV_TO_EMBEDDING_PREFIX."""

    def get_index(self, csv_basename: str):
        if self.embeddings_dir is None:
            return None
        # Try BirdNET mapping first, fall back to ARIA mapping
        prefix = BIRDNET_CSV_TO_EMBEDDING_PREFIX.get(csv_basename)
        if not prefix:
            prefix = CSV_TO_EMBEDDING_PREFIX.get(csv_basename)
        if not prefix:
            return None
        # Delegate to parent with the resolved prefix
        if prefix in self.cache:
            return self.cache[prefix]
        from feature_builder import _find_npz_for_prefix
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BioDCASE 2026 — stage-2 feature builder (BirdNET detection format)"
    )
    parser.add_argument(
        "--detections-dir",
        default=str(DEFAULT_BIRDNET_PATHS["detections_dir"]),
        help="Directory containing BirdNET detection CSVs (dev_aviary_N_detections.csv)",
    )
    parser.add_argument("--aviary-csv", default=str(DEFAULT_BIRDNET_PATHS["aviary_csv"]))
    parser.add_argument("--aviary-config", default=str(DEFAULT_BIRDNET_PATHS["aviary_config"]),
                        help="Path to aviary_config.json with day mappings for BioDCASE filename resolution")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_BIRDNET_PATHS["output_dir"] / "stage2_features_birdnet.csv"),
    )
    parser.add_argument("--bout-gap-seconds", type=float, default=6.0)
    parser.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--audio-root", action="append", default=[])
    parser.add_argument("--max-acoustic-all-segments", type=int, default=1500)
    parser.add_argument("--max-acoustic-positive-segments", type=int, default=1500)
    parser.add_argument("--max-acoustic-background-segments", type=int, default=1500)
    parser.add_argument("--maad-max-sr", type=int, default=32000)
    parser.add_argument("--maad-nperseg", type=int, default=1024)
    parser.add_argument("--maad-noverlap", type=int, default=512)
    parser.add_argument("--maad-min-segment-duration", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help=(
            "Minimum BirdNET confidence to include a detection.  Default 0.0 "
            "(include everything).  Set to e.g. 0.1 or 0.25 to filter weak "
            "predictions."
        ),
    )
    parser.add_argument(
        "--glob-pattern", default="*_detections.csv",
        help="Glob pattern for BirdNET CSV files (default: *_detections.csv)",
    )
    args = parser.parse_args()

    # ── load ground truth ──
    aviaries = parse_aviary_csv(args.aviary_csv)
    aviary_day_mappings = load_aviary_day_mappings(args.aviary_config)
    for alias, base in AVIARY_GT_ALIAS.items():
        if base in aviaries and alias not in aviaries:
            gt = aviaries[base]
            aviaries[alias] = AviaryGroundTruth(
                aviary_name=alias,
                species=dict(gt.species),
                original_names=dict(gt.original_names),
                scientific_names=dict(gt.scientific_names),
            )

    embedding_cache = BirdNETEmbeddingCache(args.embeddings_dir)

    # ── acoustic extractor (shared with ARIA pipeline) ──
    acoustic_extractor = None
    if args.audio_root:
        device = args.device
        if device == "cuda" and not TORCH_AVAILABLE:
            print(" --device cuda requested but torch+CUDA not available, falling back to CPU")
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

    # ── discover BirdNET CSVs ──
    det_dir = Path(args.detections_dir)
    csv_files = sorted(det_dir.glob(args.glob_pattern))
    if not csv_files:
        # fallback: try any CSV
        csv_files = sorted(det_dir.glob("*.csv"))

    aviary_segments = defaultdict(list)
    aviary_detections = defaultdict(list)
    aviary_source_keys = defaultdict(list)

    print("=" * 88)
    print("BIODCASE 2026 — STAGE-2 FEATURE BUILDER (BIRDNET FORMAT)")
    print("=" * 88)
    print(f"detections dir : {det_dir}")
    print(f"csv files      : {len(csv_files)}")
    print(f"glob pattern   : {args.glob_pattern}")
    print(f"min confidence : {args.min_confidence}")
    if args.embeddings_dir:
        print(f"embeddings dir : {args.embeddings_dir}")
    if args.audio_root:
        print(f"audio roots    : {args.audio_root}")
        print(f"scikit-maad    : {MAAD_AVAILABLE}")
        if acoustic_extractor:
            print(f"spectrogram    : {acoustic_extractor.device_name} (GPU={'active' if acoustic_extractor.use_gpu else 'off'})")
            print(f"workers        : {acoustic_extractor.max_workers}")
    print("=" * 88)
    print()

    unmatched_csvs = []

    for csv_file in csv_files:
        name = _strip_birdnet_suffix(csv_file.stem)

        if name not in BIRDNET_CSV_TO_AVIARY:
            unmatched_csvs.append((csv_file.name, name))
            continue

        day_mapping = aviary_day_mappings.get(name, {})
        segments, detections = parse_birdnet_csv(str(csv_file), day_mapping=day_mapping)

        # ── optional confidence filter ──
        if args.min_confidence > 0:
            before = len(detections)
            detections = [d for d in detections if d["confidence"] >= args.min_confidence]
            filtered = before - len(detections)
            if filtered > 0:
                print(f"  🔽 filtered {filtered:,d} detections below conf={args.min_confidence}")

        aviary_names = BIRDNET_CSV_TO_AVIARY[name]
        print(f"📄 {csv_file.name}  (key: {name})")
        print(f"   → {aviary_names}")
        print(f"   {len(segments):,d} segments | {len(detections):,d} detections | {compute_audio_hours(segments):.2f} audio-h")

        for aviary_name in aviary_names:
            aviary_segments[aviary_name].extend(segments)
            aviary_detections[aviary_name].extend(detections)
            aviary_source_keys[aviary_name].append(name)

    if unmatched_csvs:
        print(f"\n  {len(unmatched_csvs)} CSV(s) not matched to any aviary:")
        for fname, key in unmatched_csvs:
            print(f"   {fname}  →  extracted key: '{key}'")
        print("   → Add missing keys to BIRDNET_CSV_TO_AVIARY in this script\n")

    # ── extract features (identical pipeline to ARIA) ──
    rows = []
    print()
    print("=" * 88)
    print("EXTRACTING TARGET-SPECIES FEATURES (BIRDNET FORMAT)")
    print("=" * 88)
    print()

    aviary_iterable = [name for name in sorted(aviary_segments.keys()) if name in aviaries]
    if tqdm is not None:
        aviary_iterable = tqdm(
            aviary_iterable,
            desc="birdnet feature extraction",
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
                f"  ✓ {p['species']:22s} pop={int(p['ground_truth_count']):4d} "
                f"det/h={p['detection_rate_per_hour']:7.1f} "
                f"bout/h={p['bout_rate_per_hour']:7.1f} "
                f"occ={100*p['positive_audio_fraction']:5.1f}% "
                f"flock={p.get('flock_index', float('nan')):.2f} "
                f"fCWR={p.get('flock_corrected_cwr', float('nan')):.0f} "
                f"band={'N' if p.get('adaptive_band_source', 1.0) == 0.0 else 'W'}"
            )
        rows.extend(feats)
        print()

    save_features_csv(rows, args.output)
    print(f"\n BirdNET features saved to: {args.output}")
    print(f"   {len(rows)} rows — run the estimator with:")
    print(f"   python estimator.py --features {args.output}")
    print()


if __name__ == "__main__":
    main()
