# Healing Stones — Reconstructing Digitized Cultural Heritage Artifacts with AI

Healing Stones is a 3D reconstruction project focused on bringing fragmented cultural heritage objects back together from digital scans. In this version of the project, the pipeline works on digitized Maya stele fragments and combines geometry, classical machine learning, and robust registration to recover a plausible full assembly. The motivation is pretty simple: once fragments are scanned, we should be able to support archaeologists and conservators with reconstruction tools that are faster, more repeatable, and easier to inspect than fully manual trial and error.

## Installation

This project was developed and tested with **Python 3.12**. I recommend creating
the environment with Python 3.12 first, because Open3D was the dependency that
was most sensitive to version compatibility during setup.

```bash
pip install -r requirements.txt
```

## Usage

Standard run:

```bash
python main.py --max_vertices 100000
```

Quick iteration:

```bash
python main.py --fast
```

All reconstruction outputs, plots, and summary metrics are written to `results/`.
The same workflow can be pointed at other fragment collections through `--data_dir`, and the loader supports both `.PLY` and `.OBJ` inputs.

One practical note: **Stage 6 can take a few minutes to finish** on the standard run. That stage is where the pipeline loads the placed fragments for final export, applies the recovered transformations, writes the assembled reconstruction files, and computes the final evaluation outputs. In other words, the alignment may already be done, but the script is still doing real work while it saves the final artifact.

## Pre-trained Models

The trained surface classifier is available at: [Google Drive](https://drive.google.com/file/d/1tKKzHCmcniTnvqW-ZSkJU53fv3kVJEY0/view?usp=sharing)

Alternatively, the pipeline will retrain automatically if no saved model is found.

## Results

The latest successful reconstruction run placed all `16 / 16` fragments in a single assembly. The current semi-supervised surface classifier reports a validation accuracy of `97.2%`, pairwise RANSAC match scores generally fall in the `0.30 - 0.80` range, and the strongest refined ICP placements reach fitness values up to `0.976`.

On my local setup, the full pipeline runs in about `~75s` for the standard configuration and about `~30s` in fast mode. Those numbers are good enough for iterative experimentation while still keeping the full-resolution run practical for final output generation.

## Methodology

The pipeline starts by classifying fragment surfaces into likely break regions versus original carved or exterior surfaces. I use a semi-supervised Random Forest for this step because the dataset does not come with manual point-level labels, so the classifier has to bootstrap itself from geometric heuristics before being refined into a more stable model. That classification step is mostly a search-space reduction stage: if the break surfaces are isolated reasonably well, the later matching stages become much more reliable.

For matching, the core descriptor is FPFH, but an important detail is that the descriptor scale is adapted per fragment rather than fixed globally. Some fragments are tiny chips while others are large slab-like sections, so a single voxel size does not make sense across the whole dataset. After coarse matching, the pipeline uses multi-scale RANSAC and a robust ICP refinement step with a Tukey loss kernel so that erosion gaps and noisy correspondences do not completely derail the alignment.

The final assembly is built greedily from post-ICP candidate edges, with additional quality gates applied after refinement. That design turned out to be a good balance between something I could implement and reason about during the project and something strong enough to produce a full 16-fragment reconstruction.

## Known Limitations

- Large-fragment pairwise RMSE is still relatively high in the `5-10 mm` range.
- Global assembly can accumulate drift because the current method is still greedy rather than globally optimized.
- The current `82%` gap estimate should be interpreted as a sign of alignment imprecision, not as a literal physical void measurement.

## Future Work

This is the part I would most like to push further as a GSoC project:

- Pose graph optimization for better global consistency after the greedy assembly step
- Learned registration models such as DCP or PRNet for gap-tolerant fragment alignment
- A PointNet-style surface classifier trained with a small manually annotated set of fragments
- Bundle adjustment across all placed fragments so local good fits do not accumulate into global drift

## Repository Layout

- `main.py` — end-to-end pipeline entry point
- `src/data_loader.py` — mesh loading, centering, and preprocessing
- `src/surface_classifier.py` — semi-supervised surface labeling
- `src/feature_extraction.py` — adaptive geometric descriptors
- `src/matching.py` — pairwise fragment matching and scoring
- `src/alignment.py` — ICP refinement and global assembly
- `src/visualization.py` — plots, screenshots, and interactive exports
- `notebooks/exploration.py` — exploratory notebook in percent format

## Acknowledgments

This project was prepared in the spirit of a GSoC 2026 proposal and would not exist without the broader digital heritage and AI mentoring context around it. I especially want to acknowledge the inspiration and support of CERN HumanAI and mentors connected with the University of Alabama.
