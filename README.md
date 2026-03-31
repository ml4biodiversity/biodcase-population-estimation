# BioDCASE 2026 — Population Estimation Baseline

Baseline system for the **BioDCASE 2026 Challenge: Avian Population Estimation from Passive Acoustic Recordings**.

Given continuous audio recordings from zoo aviaries, the task is to estimate the number of individuals of each target species present in the aviary. This repository provides the complete baseline pipeline: from detection CSVs to population estimates.

## Challenge overview

Participants receive audio recordings from **6 aviaries** (`aviary_1` through `aviary_6`). Each aviary contains multiple bird species, but population estimation is evaluated only for the **target species**:

| Species | Aviaries |
|---|---|
| Greater flamingo (*Phoenicopterus roseus*) | aviary_2, aviary_4, aviary_5, aviary_6 |
| Red-billed quelea (*Quelea quelea*) | aviary_1, aviary_3 |
| Hadada ibis (*Bostrychia hagedash*) | aviary_2, aviary_4 |

Ground truth populations range from 4 to 161 individuals. The full species inventory for each aviary (including non-target species) is provided in `ground_truth.csv`.

## Pipeline

The baseline operates as a two-stage pipeline:

```
Audio recordings
    │
    ▼
┌─────────────────────┐
│  Species Detection   │   pip install aria-inference
│  (ARIA or BirdNET)   │   pip install aria-inference-birdnet
└─────────┬───────────┘
          │  detection CSVs
          ▼
┌─────────────────────┐
│  Feature Builder     │   feature_builder.py / feature_builder_birdnet.py
│  (Stage 2)           │
└─────────┬───────────┘
          │  feature CSV
          ▼
┌─────────────────────┐
│  Population          │   estimator.py
│  Estimator           │
└─────────────────────┘
          │
          ▼
    Population estimates
```

### Stage 1: Species detection

Run a bird species detector on each aviary's audio files. Two detection packages are supported:

**Option A — ARIA** (recommended baseline):
```bash
pip install aria-inference
aria-inference --input /path/to/aviary_1_audio/ --output detections/aviary_1_detections.csv
```

**Option B — BirdNET**:
```bash
pip install aria-inference-birdnet
aria-inference-birdnet --input /path/to/aviary_1_audio/ --output detections/aviary_1_detections.csv
```

Detection CSVs must be named `aviary_N_detections.csv` and placed in the `detections/` directory.

**ARIA detection format:**
```
File,Segment,Start,End,Species,Confidence,Method,Status
```

**BirdNET detection format:**
```
Start (s),End (s),Scientific name,Common name,Confidence,File
```

### Stage 2: Feature extraction

The feature builder reads detection CSVs and extracts a rich feature vector per (aviary, target-species) pair. Features include detection-count statistics, temporal bout structure, and optionally scikit-maad acoustic indices.

**For ARIA detections:**
```bash
python feature_builder.py \
    --detections-dir detections/ \
    --output features/stage2_features.csv
```

**For BirdNET detections:**
```bash
python feature_builder_birdnet.py \
    --detections-dir detections/ \
    --output features/stage2_features_birdnet.csv
```

**Optional: acoustic features** (requires access to the original audio and `scikit-maad`):
```bash
pip install scikit-maad

python feature_builder.py \
    --detections-dir detections/ \
    --audio-root /path/to/audio/ \
    --device cuda --workers 4 \
    --output features/stage2_features.csv
```

### Stage 3: Population estimation

The estimator fits species-specific leave-one-out regression models and reports per-aviary population estimates:

```bash
python estimator.py --features features/stage2_features.csv
```

## Repository structure

```
biodcase-population-estimation/
├── README.md                    # This file
├── ground_truth.csv             # Species inventory and populations (all species, 6 aviaries)
├── aviary_config.json           # Aviary metadata (recording days, target species, file counts)
├── feature_builder.py           # Stage-2 feature builder (ARIA detection format)
├── feature_builder_birdnet.py   # Stage-2 feature builder (BirdNET detection format)
├── estimator.py                 # Population estimator (stage 2)
├── detections/                  # Place detection CSVs here (not included)
└── features/                    # Output directory for feature CSVs (created automatically)
```

## Dependencies

Core (required):
```
numpy
scipy
soundfile
```

Optional:
```
scikit-maad          # Acoustic index features
torch                # GPU-accelerated spectrograms (--device cuda)
tqdm                 # Progress bars
```

Install all:
```bash
pip install numpy scipy soundfile scikit-maad torch tqdm
```

## Key features extracted

The feature builder extracts **80+ features** per (aviary, species) pair, organized into blocks:

**Detection statistics** — detection rate, confidence-weighted rate (CWR), mean/std confidence

**Temporal structure** — bout count, bout rate, bout duration, inter-bout gaps, active hours, temporal spread

**Occupancy** — positive segment fraction, bout span fraction, active file fraction

**Co-occurrence** — overlap with other species, species labels per segment, confidence margin

**Acoustic indices** (optional, requires `--audio-root`) — scikit-maad indices (ACI, NDSI, BI, Hf, nROI, etc.) computed separately for target-positive, background, and scene-wide segments, plus contrast features and 5 composite scores

**Flock-calling indicators** — energy stability, event suppression, spectral persistence, background bleed, composite flock index, flock-corrected CWR

**Adaptive band selection** — automatic narrow vs. wide frequency band selection based on positive-minus-background contrast

## Ground truth format

`ground_truth.csv` contains all bird species present in each aviary, not just target species:

```csv
aviary_id,common_name,scientific_name,count,is_target
aviary_1,Red-billed quelea,Quelea quelea,153,1
aviary_1,African spoonbill,Platalea alba,9,0
...
```

The `is_target` column indicates which species are evaluated for population estimation.

## Detection CSV naming

Detection CSV filenames must follow the pattern `aviary_N_detections.csv`:

```
detections/
├── aviary_1_detections.csv
├── aviary_2_detections.csv
├── aviary_3_detections.csv
├── aviary_4_detections.csv
├── aviary_5_detections.csv
└── aviary_6_detections.csv
```

## Audio file format

Audio recordings are organised as:

```
biodcase_2026/
├── aviary_1/
│   ├── chunk_000/
│   │   ├── rec_d1_00_00_45.750000.wav
│   │   ├── rec_d1_00_01_49.wav
│   │   └── ...
│   ├── chunk_001/
│   │   └── ...
│   └── ...
├── aviary_2/
│   └── ...
└── ...
```

Filenames follow the pattern `rec_dN_HH_MM_SS[.ffffff].wav` where `dN` is a day identifier (d1, d2, d3) and the rest encodes the time of day. The mapping from day identifiers to calendar dates is in `aviary_config.json`. The feature builder loads this mapping automatically via `--aviary-config` to reconstruct full timestamps for temporal features.

## Notes

- **Aviary 5 and aviary 6** are two separate recording sessions from the same physical location with the same bird population. They are treated as independent data points with different acoustic conditions.

- **Acoustic features are optional.** The detection-only feature set (without `--audio-root`) already captures the primary signal for species where detection-count regression works well (Red-billed quelea, Hadada ibis). Acoustic features provide marginal improvements and are most relevant for flock-calling species (Greater flamingo) where detection counts underestimate population due to synchronized vocalizations.

- **The estimator uses leave-one-out cross-validation** since the dataset contains only 6 aviaries. It runs 5 model families and reports the best per species. Three models (1, 2, 4) work with detection-only features; two models (3, 5) additionally require acoustic features from `--audio-root`. If acoustic features are absent, models 3 and 5 are automatically skipped.

## Citation

If you use this repository, please cite the baseline software.
@software{argin2026biodcase_baseline,
  author       = {Arg{\i}n, Emre and H{\"a}rm{\"a}, Aki and Arslan-Dogan, Aysenur},
  title        = {{BioDCASE 2026 Bird Counting Baseline: Avian Population Estimation from Passive Acoustic Recordings}},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ml4biodiversity/biodcase-population-estimation},
  version      = {1.0.0},
}

If you use the **BioDCASE 2026 Bird Counting** dataset, please also cite the dataset.
@dataset{argin2026biodcase_dataset,
  author       = {Arg{\i}n, Emre and H{\"a}rm{\"a}, Aki and Arslan-Dogan, Aysenur},
  title        = {{BioDCASE 2026 Bird Counting: Avian Population Estimation from Passive Acoustic Recordings}},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/Emreargin/BioDCASE2026_Bird_Counting},
}

If you use **ARIA** detections or build on the ARIA methodology, please also cite the ARIA paper.
@inproceedings{argin2026aria,
  author       = {Arg{\i}n, Emre and H{\"a}rm{\"a}, Aki and Dreesen, Philippe},
  title        = {{ARIA: Acoustic Recognition for Inventory in Aviaries}},
  booktitle    = {Proceedings of the IEEE World Congress on Computational Intelligence (WCCI) / International Joint Conference on Neural Networks (IJCNN)},
  year         = {2026},
  note         = {Accepted, to appear},
}

## License

MIT