"""End-to-end entry point for the healing-stones reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import open3d as o3d

import src.matching as matching_module
from src.alignment import (
    PAIR_REFINED_FITNESS_THRESHOLD,
    align_pair,
    assemble_reconstruction,
    global_registration,
    multi_scale_icp,
)
from src.config import FAST_MAX_VERTICES, LOGGER, MODELS_DIR, WORKING_MAX_VERTICES
from src.data_loader import (
    load_all_fragments,
    load_selected_full_meshes,
)
from src.feature_extraction import extract_all_features
from src.matching import filter_matches, find_all_matches, predict_adjacency_ml
from src.surface_classifier import (
    compute_surface_features,
    extract_break_surface,
    generate_pseudolabels,
    refine_with_confidence,
    train_classifier_from_pseudolabels,
    transfer_labels_to_mesh,
)
from src.visualization import (
    export_reconstruction_plotly,
    plot_match_matrix,
    plot_match_scores,
    plot_metrics_summary,
    visualize_matches,
    visualize_reconstruction,
    visualize_surface_classification,
)


DEFAULT_MATCH_THRESHOLD = 0.3  # 0.3 seemed to work best after testing a few values
DEFAULT_VOXEL_SCALES = [2.0, 1.0, 0.5]
DEFAULT_PLOTLY_POINTS = 25000
EPSILON = 1e-8


class RawDefaultsFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve example formatting while still showing default values."""


def _paths_alias_same_file(source: Path, destination: Path) -> bool:
    """Return True when two paths already reference the same on-disk file."""
    try:
        return os.path.samefile(source, destination)
    except FileNotFoundError:
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a fragmented cultural heritage artifact from 3D mesh scans "
            "using surface classification, adaptive feature matching, and robust ICP."
        ),
        epilog=(
            "Examples:\n"
            "  python main.py --max_vertices 100000\n"
            "  python main.py --fast"
        ),
        formatter_class=RawDefaultsFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="data/fragments/",
        help="Directory containing input PLY/OBJ fragments",
    )
    parser.add_argument(
        "--output_dir",
        default="results/",
        help="Directory where plots, metrics, and reconstructed meshes are written",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.5,
        help="Base preprocessing voxel size; descriptor scales adapt per fragment",
    )
    parser.add_argument(
        "--use_ml",
        action="store_true",
        help="Enable the experimental learned pair-ranking module",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open lightweight interactive previews and debug plots during the run",
    )
    parser.add_argument(
        "--skip_classification",
        action="store_true",
        help="Use full fragment surfaces instead of predicted break surfaces",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller working mesh budget for quicker iteration",
    )
    parser.add_argument(
        "--max_vertices",
        type=int,
        default=WORKING_MAX_VERTICES,
        help="Target working point budget for each fragment cloud",
    )
    return parser.parse_args()


def _start_stage(name: str) -> float:
    """Log the start of a pipeline stage and return a timer."""
    LOGGER.info("")
    LOGGER.info("=== %s ===", name)
    return time.perf_counter()


def _finish_stage(name: str, started_at: float, stage_times: Dict[str, float]) -> None:
    """Record and log the runtime of a pipeline stage."""
    elapsed = time.perf_counter() - started_at
    stage_times[name] = elapsed
    LOGGER.info("%s finished in %.2f seconds", name, elapsed)


def _load_surface_model() -> Any | None:
    """Load a saved surface-classification model if one exists."""
    model_paths = [
        MODELS_DIR / "surface_classifier.pkl",
        MODELS_DIR / "surface_classifier.joblib",
    ]
    for model_path in model_paths:
        if not model_path.exists():
            continue

        try:
            model = joblib.load(model_path)
            LOGGER.info("Loaded trained surface-classifier model from %s", model_path)
            validation_accuracy = getattr(model, "validation_accuracy_", None)
            if validation_accuracy is not None:
                LOGGER.info(
                    "Surface classifier stored validation accuracy: %.4f",
                    float(validation_accuracy),
                )
            return model
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Could not load %s, trying the next fallback: %s",
                model_path,
                exc,
            )

    LOGGER.info(
        "No trained surface-classifier model found, bootstrapping from pseudo-labels"
    )
    return None


def _mesh_vertices_to_pointcloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    """Build a point cloud directly from mesh vertices."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    return pcd


def _copy_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Return a shallow copy of a point cloud."""
    return o3d.geometry.PointCloud(pcd)


def _short_fragment_name(name: str) -> str:
    """Generate compact labels for console score matrices."""
    if "FR_" in name:
        try:
            return f"FR_{int(name.split('FR_')[1][:2]):02d}"
        except (ValueError, IndexError):
            pass
    return name[:10]


def _build_score_matrix(
    fragment_names: List[str],
    matches: List[Dict[str, Any]],
    score_key: str = "score",
) -> np.ndarray:
    """Build a symmetric score matrix for plotting."""
    matrix = np.full((len(fragment_names), len(fragment_names)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    index_lookup = {name: idx for idx, name in enumerate(fragment_names)}

    for match in matches:
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        score = float(match.get(score_key, np.nan))
        if name_a not in index_lookup or name_b not in index_lookup:
            continue
        i = index_lookup[name_a]
        j = index_lookup[name_b]
        matrix[i, j] = score
        matrix[j, i] = score
    return matrix


def _merge_ml_scores(
    all_matches: List[Dict[str, Any]],
    ml_predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine experimental ML scores with the primary RANSAC scores."""
    if not ml_predictions:
        return all_matches

    ml_scores = {}
    for prediction in ml_predictions:
        key = tuple(sorted((prediction["fragment_a"], prediction["fragment_b"])))
        ml_scores[key] = float(prediction["adjacency_score"])

    merged: List[Dict[str, Any]] = []
    for match in all_matches:
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        base_score = float(match["score"])
        key = tuple(sorted((name_a, name_b)))
        ml_score = ml_scores.get(key, base_score)
        combined_score = 0.7 * float(base_score) + 0.3 * float(ml_score)
        merged_match = dict(match)
        merged_match["score"] = float(combined_score)
        merged_match["base_score"] = base_score
        merged.append(merged_match)
    return merged


def _print_score_matrix(
    title: str,
    fragment_names: List[str],
    matches: List[Dict[str, Any]],
    score_key: str,
) -> None:
    """Print a symmetric matrix so we can compare pre/post ICP quality quickly."""
    matrix = _build_score_matrix(fragment_names, matches, score_key=score_key)
    short_names = [_short_fragment_name(name) for name in fragment_names]
    header = "fragment".ljust(14) + "".join(name.rjust(12) for name in short_names)
    print(f"\n{title}")
    print(header)
    for row_name, row_values in zip(short_names, matrix):
        cells = []
        for value in row_values:
            if np.isnan(value):
                cells.append("    --     ")
            else:
                cells.append(f"{value:12.3f}")
        print(row_name.ljust(14) + "".join(cells))


def _jsonify(value: Any) -> Any:
    """Convert numpy/Open3D-heavy objects into JSON-safe structures."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file with light numpy conversion."""
    path.write_text(json.dumps(_jsonify(payload), indent=2))


def _copy_output_geometry(geometry: o3d.geometry.TriangleMesh | o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh | o3d.geometry.PointCloud:
    """Return a copy of a mesh or point cloud for safe transformations."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        return o3d.geometry.TriangleMesh(geometry)
    if isinstance(geometry, o3d.geometry.PointCloud):
        return o3d.geometry.PointCloud(geometry)
    raise TypeError("Unsupported output geometry type")


def _geometry_points(geometry: o3d.geometry.TriangleMesh | o3d.geometry.PointCloud) -> np.ndarray:
    """Extract xyz points from a mesh or point cloud."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        return np.asarray(geometry.vertices)
    if isinstance(geometry, o3d.geometry.PointCloud):
        return np.asarray(geometry.points)
    raise TypeError("Unsupported output geometry type")


def _transform_geometries(
    geometries: Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud],
    transformations: Dict[str, np.ndarray],
) -> Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud]:
    """Apply transformations to meshes or point clouds used in Stage 6."""
    transformed: Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud] = {}
    for name, transform in transformations.items():
        if name not in geometries:
            continue
        geometry_copy = _copy_output_geometry(geometries[name])
        geometry_copy.transform(transform)
        if isinstance(geometry_copy, o3d.geometry.TriangleMesh):
            geometry_copy.compute_vertex_normals()
        transformed[name] = geometry_copy
    return transformed


def _transform_fragments(
    fragments: Dict[str, o3d.geometry.TriangleMesh],
    transformations: Dict[str, np.ndarray],
) -> Dict[str, o3d.geometry.TriangleMesh]:
    """Apply final transformations to fragment meshes for visualization."""
    transformed: Dict[str, o3d.geometry.TriangleMesh] = {}
    for name, transform in transformations.items():
        mesh_copy = o3d.geometry.TriangleMesh(fragments[name])
        mesh_copy.transform(transform)
        mesh_copy.compute_vertex_normals()
        transformed[name] = mesh_copy
    return transformed


def _prepare_stage6_meshes(
    fragment_metadata: Dict[str, Dict[str, object]],
    fragment_names: List[str],
) -> Dict[str, o3d.geometry.TriangleMesh]:
    """Load only the meshes needed for Stage 6."""
    fragment_paths = {
        name: Path(metadata["path"])
        for name, metadata in fragment_metadata.items()
    }
    return load_selected_full_meshes(fragment_paths, fragment_names)


def _sample_reconstruction_points(
    fragments: Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud],
    transformations: Dict[str, np.ndarray],
    total_points: int = DEFAULT_PLOTLY_POINTS,
) -> np.ndarray:
    """Build a lightweight point sample for interactive exports without resampling the full mesh."""
    if total_points <= 0 or not fragments or not transformations:
        return np.empty((0, 3), dtype=float)

    placed_names = [name for name in transformations if name in fragments]
    if not placed_names:
        return np.empty((0, 3), dtype=float)

    per_fragment = max(total_points // len(placed_names), 256)
    sampled_sets: List[np.ndarray] = []

    for name in placed_names:
        geometry = fragments[name]
        points = _geometry_points(geometry)
        if len(points) == 0:
            continue

        if len(points) > per_fragment:
            indices = np.linspace(0, len(points) - 1, per_fragment, dtype=int)
            points = points[indices]

        homogenous = np.hstack([points, np.ones((len(points), 1), dtype=float)])
        transformed = (np.asarray(transformations[name], dtype=float) @ homogenous.T).T[:, :3]
        sampled_sets.append(transformed)

    if not sampled_sets:
        return np.empty((0, 3), dtype=float)

    points = np.vstack(sampled_sets)
    if len(points) > total_points:
        indices = np.linspace(0, len(points) - 1, total_points, dtype=int)
        points = points[indices]
    return points


def _offset_assembly_groups(
    fragments: Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud],
    transformations: Dict[str, np.ndarray],
    assembly_groups: List[List[str]],
    spacing: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Offset disconnected assemblies so they render as separate groups."""
    offset_transforms = {
        name: np.array(transform, dtype=float, copy=True)
        for name, transform in transformations.items()
    }
    if len(assembly_groups) <= 1:
        return offset_transforms

    max_diagonal = max(
        (
            float(np.linalg.norm(fragments[name].get_axis_aligned_bounding_box().get_extent()))
            for name in offset_transforms
            if name in fragments
        ),
        default=0.0,
    )
    spacing = max(float(spacing), max_diagonal * 0.1)
    running_max_x: float | None = None

    for index, group in enumerate(assembly_groups):
        group_names = [name for name in group if name in offset_transforms and name in fragments]
        if not group_names:
            continue

        group_min_x = float("inf")
        group_max_x = float("-inf")
        for name in group_names:
            geometry_copy = _copy_output_geometry(fragments[name])
            geometry_copy.transform(offset_transforms[name])
            bbox = geometry_copy.get_axis_aligned_bounding_box()
            group_min_x = min(group_min_x, float(bbox.min_bound[0]))
            group_max_x = max(group_max_x, float(bbox.max_bound[0]))

        if index == 0 or running_max_x is None:
            running_max_x = group_max_x
            continue

        translate_x = (running_max_x + spacing) - group_min_x
        if translate_x > 0:
            translation = np.eye(4)
            translation[0, 3] = translate_x
            for name in group_names:
                offset_transforms[name] = translation @ offset_transforms[name]
            group_max_x += translate_x

        running_max_x = max(running_max_x, group_max_x)

    return offset_transforms


def _assembly_bbox_volume(
    fragments: Dict[str, o3d.geometry.TriangleMesh | o3d.geometry.PointCloud],
    transformations: Dict[str, np.ndarray],
    group_names: List[str],
) -> float:
    """Measure the occupied bounding-box volume of one assembly group."""
    group_geometries = []
    for name in group_names:
        if name not in fragments or name not in transformations:
            continue
        geometry_copy = _copy_output_geometry(fragments[name])
        geometry_copy.transform(transformations[name])
        group_geometries.append(geometry_copy)

    if not group_geometries:
        return 0.0

    mins = []
    maxs = []
    for geometry in group_geometries:
        bbox = geometry.get_axis_aligned_bounding_box()
        mins.append(np.asarray(bbox.min_bound, dtype=float))
        maxs.append(np.asarray(bbox.max_bound, dtype=float))

    extent = np.maximum(np.max(maxs, axis=0) - np.min(mins, axis=0), EPSILON)
    return float(np.prod(extent))


def _format_metric(value: Any) -> str:
    """Format metrics for the final summary table."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_duration(seconds: float) -> str:
    """Format a duration in a compact human-readable form."""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _log_eta(stage_name: str, processed: int, total: int, started_at: float) -> None:
    """Log a simple ETA based on average per-item stage time so far."""
    if processed <= 0 or processed >= total:
        return

    elapsed = time.perf_counter() - started_at
    average_time = elapsed / float(processed)
    eta_seconds = average_time * float(total - processed)
    LOGGER.info(
        "%s progress %d/%d, estimated time remaining: %s",
        stage_name,
        processed,
        total,
        _format_duration(eta_seconds),
    )


def main() -> None:
    """Run the full Healing Stones reconstruction workflow."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_times: Dict[str, float] = {}
    fragment_issues: Dict[str, List[str]] = defaultdict(list)
    total_start = time.perf_counter()

    LOGGER.info("Starting Healing Stones reconstruction pipeline")
    LOGGER.info("Data directory: %s", Path(args.data_dir).resolve())
    LOGGER.info("Output directory: %s", output_dir.resolve())
    max_vertices = FAST_MAX_VERTICES if args.fast else args.max_vertices
    use_ml_matching = args.use_ml and not args.fast
    if args.fast and args.use_ml:
        LOGGER.info("Fast mode enabled, so experimental ML matching will be skipped")

    # Stage 1 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 1: Data Loading")
    meshes_full, meshes_working, fragment_metadata = load_all_fragments(
        args.data_dir,
        max_vertices=max_vertices,
        load_full_meshes=not args.fast,
    )
    fragment_names = sorted(meshes_working.keys())

    if not fragment_names:
        LOGGER.error("No fragments were loaded. Exiting.")
        return

    LOGGER.info("Working point budget per fragment: %d points", max_vertices)

    if args.visualize:
        # One quick preview is enough to confirm the data loaded correctly.
        first_name = fragment_names[0]
        try:
            from src.visualization import visualize_fragment

            preview_meshes = meshes_full
            if first_name not in preview_meshes and first_name in fragment_metadata:
                preview_meshes = load_selected_full_meshes(
                    {first_name: Path(fragment_metadata[first_name]["path"])},
                    [first_name],
                )

            visualize_fragment(
                preview_meshes[first_name],
                title=f"Fragment {first_name}",
                output_dir=output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Fragment preview failed for %s: %s", first_name, exc)

    _finish_stage("Stage 1: Data Loading", stage_start, stage_times)

    # Stage 2 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 2: Surface Classification")
    surface_model = None if args.skip_classification else _load_surface_model()
    break_surfaces: Dict[str, Tuple[o3d.geometry.PointCloud, np.ndarray]] = {}
    classification_summary: Dict[str, Dict[str, Any]] = {}
    surface_breakdown = {"break": 0, "original": 0, "uncertain": 0}
    feature_map: Dict[str, np.ndarray] = {}
    pseudolabel_map: Dict[str, np.ndarray] = {}
    pseudoconfidence_map: Dict[str, np.ndarray] = {}
    feature_ready_names: List[str] = []

    if not args.skip_classification:
        for index, name in enumerate(fragment_names):
            working_pcd = meshes_working[name]
            try:
                features = compute_surface_features(
                    mesh=None,
                    pcd=working_pcd,
                    progress_desc=f"{name[:18]} surface",
                )
                feature_map[name] = features
                feature_ready_names.append(name)

                if surface_model is None:
                    pseudolabels, pseudoconfidence = generate_pseudolabels(features)
                    pseudolabel_map[name] = pseudolabels
                    pseudoconfidence_map[name] = pseudoconfidence

                _log_eta("Stage 2", index + 1, len(fragment_names), stage_start)
            except Exception as exc:  # noqa: BLE001
                message = f"surface feature extraction failed: {exc}"
                fragment_issues[name].append(message)
                LOGGER.warning("%s (%s)", message, name)

        if surface_model is None and pseudolabel_map:
            all_features = np.vstack([feature_map[name] for name in feature_ready_names])
            all_pseudolabels = np.concatenate([pseudolabel_map[name] for name in feature_ready_names])
            try:
                LOGGER.info(
                    "Training semi-supervised surface classifier from %d pseudo-labeled points",
                    len(all_pseudolabels),
                )
                surface_model = train_classifier_from_pseudolabels(
                    all_features,
                    all_pseudolabels,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Could not train the surface classifier from pseudo-labels, falling back to k-means labels: %s",
                    exc,
                )

    for index, name in enumerate(fragment_names):
        working_pcd = meshes_working[name]
        full_mesh = meshes_full.get(name)
        try:
            if args.skip_classification:
                labels = np.ones(len(working_pcd.points), dtype=int)
                confidence = np.ones(len(working_pcd.points), dtype=float)
                break_pcd = _copy_pointcloud(working_pcd)
                break_indices = np.arange(len(working_pcd.points), dtype=int)
            else:
                if name not in feature_map:
                    raise ValueError("surface features were not computed for this fragment")

                features = feature_map[name]
                if surface_model is not None and name in pseudolabel_map:
                    labels, confidence = refine_with_confidence(
                        surface_model,
                        features,
                        pseudolabel_map[name],
                    )
                elif surface_model is not None:
                    labels = surface_model.predict(features).astype(int)
                    if hasattr(surface_model, "predict_proba"):
                        confidence = surface_model.predict_proba(features).max(axis=1)
                    else:
                        confidence = np.ones(len(labels), dtype=float)
                else:
                    labels = pseudolabel_map[name]
                    confidence = pseudoconfidence_map[name]

                break_pcd, break_indices = extract_break_surface(
                    mesh=None,
                    pcd=working_pcd,
                    labels=labels,
                )

                if break_pcd.is_empty():
                    LOGGER.warning(
                        "Break-surface extraction returned no points for %s, using all working points",
                        name,
                    )
                    break_pcd = _copy_pointcloud(working_pcd)
                    break_indices = np.arange(len(working_pcd.points), dtype=int)
                    labels = np.ones(len(working_pcd.points), dtype=int)
                    confidence = np.ones(len(working_pcd.points), dtype=float)

            uncertain_count = int(np.sum(confidence < 0.6)) if confidence is not None else 0
            break_count = int(np.sum(labels == 1))
            original_count = int(np.sum(labels == 0))

            classification_summary[name] = {
                "full_mesh_vertices": int(fragment_metadata[name]["original_vertices"]),
                "working_points": int(len(working_pcd.points)),
                "break_points": break_count,
                "original_points": original_count,
                "uncertain_points": uncertain_count,
                "mean_confidence": float(np.mean(confidence))
                if confidence is not None
                else None,
            }
            surface_breakdown["break"] += break_count
            surface_breakdown["original"] += original_count
            surface_breakdown["uncertain"] += uncertain_count
            break_surfaces[name] = (break_pcd, break_indices)

            LOGGER.info(
                "%s -> break=%d, original=%d, uncertain=%d",
                name,
                break_count,
                original_count,
                uncertain_count,
            )

            if (
                args.visualize
                and index == 0
                and not args.skip_classification
                and full_mesh is not None
            ):
                full_labels, full_confidence = transfer_labels_to_mesh(
                    full_mesh,
                    working_pcd,
                    labels,
                    confidence,
                )
                visualize_surface_classification(
                    full_mesh,
                    (full_labels, full_confidence),
                    title=f"Surface Classification: {name}",
                    output_dir=output_dir,
                )
        except Exception as exc:  # noqa: BLE001
            message = f"surface classification failed: {exc}"
            fragment_issues[name].append(message)
            LOGGER.warning("%s (%s)", message, name)

    _finish_stage("Stage 2: Surface Classification", stage_start, stage_times)

    # Stage 3 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 3: Feature Extraction")
    features_dict: Dict[str, Dict[str, Any]] = {}
    feature_times: Dict[str, float] = {}

    for name in fragment_names:
        if name not in break_surfaces:
            continue

        break_surface_pcd = break_surfaces[name][0]
        feature_start = time.perf_counter()
        try:
            features_dict[name] = extract_all_features(
                break_surface_pcd,
                voxel_size=args.voxel_size,
                fragment_name=name,
            )
            feature_times[name] = time.perf_counter() - feature_start
            LOGGER.info("%s feature extraction took %.2f seconds", name, feature_times[name])
            _log_eta("Stage 3", len(feature_times), len(fragment_names), stage_start)
        except Exception as exc:  # noqa: BLE001
            message = f"feature extraction failed: {exc}"
            fragment_issues[name].append(message)
            LOGGER.warning("%s (%s)", message, name)

    if feature_times:
        LOGGER.info("Mean feature extraction time: %.2f seconds", float(np.mean(list(feature_times.values()))))
    # TODO: Save per-fragment feature timings in a nicer CSV report once the pipeline stabilizes.

    _finish_stage("Stage 3: Feature Extraction", stage_start, stage_times)

    # Stage 4 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 4: Matching")
    all_matches = find_all_matches(
        meshes_working,
        break_surfaces,
        features_dict,
        score_threshold=0.0,
    )

    experimental_ml_matches: List[Dict[str, Any]] = []
    combined_matches = [
        match for match in all_matches if float(match["score"]) >= DEFAULT_MATCH_THRESHOLD
    ]

    if use_ml_matching:
        try:
            matching_module.ENABLE_EXPERIMENTAL_ML = True
            experimental_ml_matches = predict_adjacency_ml(
                meshes_working,
                break_surfaces,
                features_dict,
            )
            combined_matches = _merge_ml_scores(all_matches, experimental_ml_matches)
            combined_matches = [
                match
                for match in combined_matches
                if float(match["score"]) >= DEFAULT_MATCH_THRESHOLD
            ]
            LOGGER.info("Experimental ML matching produced %d predictions", len(experimental_ml_matches))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Experimental ML matching failed, keeping pure RANSAC scores: %s", exc)
        finally:
            matching_module.ENABLE_EXPERIMENTAL_ML = False

    filtered_matches = filter_matches(combined_matches, max_matches_per_fragment=3)
    LOGGER.info("Kept %d filtered pairwise matches", len(filtered_matches))

    match_matrix = _build_score_matrix(fragment_names, all_matches)
    plot_match_matrix(match_matrix, fragment_names, output_dir=output_dir)
    plot_match_scores(
        [
            {
                "fragment_a": str(match["fragment_a"]),
                "fragment_b": str(match["fragment_b"]),
                "score": float(match["score"]),
            }
            for match in filtered_matches
        ],
        output_dir=output_dir,
    )

    matches_payload = {
        "all_matches": all_matches,
        "filtered_matches": filtered_matches,
        "experimental_ml_matches": experimental_ml_matches,
    }
    _save_json(output_dir / "matches.json", matches_payload)
    _finish_stage("Stage 4: Matching", stage_start, stage_times)

    # Stage 5 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 5: Alignment")
    refined_matches: List[Dict[str, Any]] = []
    pair_alignment_rmse: Dict[str, float] = {}
    gap_distances: List[float] = []

    for index, match in enumerate(filtered_matches):
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        score = float(match["score"])
        transform = np.asarray(match["transformation"], dtype=float)
        try:
            source_pcd = break_surfaces[name_a][0]
            target_pcd = break_surfaces[name_b][0]

            # We keep the RANSAC transform as the initializer because coarse
            # matching is usually much more reliable than starting ICP cold.
            refined = multi_scale_icp(
                source_pcd,
                target_pcd,
                initial_transform=transform,
                voxel_sizes=sorted(set(DEFAULT_VOXEL_SCALES + [args.voxel_size]), reverse=True),
                rejection_threshold=PAIR_REFINED_FITNESS_THRESHOLD,
            )
            refined_record = dict(match)
            refined_record["transformation"] = np.asarray(refined["transformation"], dtype=float)
            refined_record["refined_fitness"] = float(refined["fitness"])
            refined_record["refined_inlier_rmse"] = float(refined["inlier_rmse"])
            refined_record["refined_correspondence_count"] = int(refined["correspondence_count"])
            refined_record["rejected"] = bool(refined.get("rejected", False))
            refined_matches.append(refined_record)

            if refined_record["rejected"]:
                LOGGER.info(
                    "REJECTED: %s vs %s, refined fitness %.3f < threshold %.2f",
                    _short_fragment_name(name_a),
                    _short_fragment_name(name_b),
                    float(refined_record["refined_fitness"]),
                    PAIR_REFINED_FITNESS_THRESHOLD,
                )
                continue

            pair_key = f"{name_a}__{name_b}"
            pair_alignment_rmse[pair_key] = float(refined["inlier_rmse"])

            _, gap_stats = align_pair(
                meshes_working[name_a],
                meshes_working[name_b],
                np.asarray(refined["transformation"], dtype=float),
            )
            gap_distances.append(float(gap_stats["mean_distance"]))

            LOGGER.info(
                "%s vs %s -> fitness=%.3f, rmse=%.4f",
                name_a,
                name_b,
                refined["fitness"],
                refined["inlier_rmse"],
            )

            if args.visualize and index == 0:
                match_meshes = meshes_full
                if name_a not in match_meshes or name_b not in match_meshes:
                    match_meshes = {
                        **match_meshes,
                        **load_selected_full_meshes(
                            {
                                name_a: Path(fragment_metadata[name_a]["path"]),
                                name_b: Path(fragment_metadata[name_b]["path"]),
                            },
                            [name_a, name_b],
                        ),
                    }
                visualize_matches(
                    match_meshes[name_a],
                    match_meshes[name_b],
                    refined["transformation"],
                    title=f"Match: {name_a} vs {name_b}",
                    output_dir=output_dir,
                )
        except Exception as exc:  # noqa: BLE001
            message = f"alignment refinement failed: {exc}"
            fragment_issues[name_a].append(message)
            fragment_issues[name_b].append(message)
            LOGGER.warning("Failed to refine %s vs %s: %s", name_a, name_b, exc)

    _print_score_matrix(
        "Post-ICP Refined Matrix",
        fragment_names,
        refined_matches,
        score_key="refined_fitness",
    )

    break_surface_sizes = {
        name: int(len(break_surfaces[name][0].points))
        for name in break_surfaces
    }
    registration_result = global_registration(
        meshes_working,
        refined_matches,
        break_surface_sizes=break_surface_sizes,
        break_surfaces=break_surfaces,
        features_dict=features_dict,
    )
    transformations = {
        name: np.asarray(transform, dtype=float)
        for name, transform in registration_result.get("transformations", {}).items()
    }
    assembly_groups = [
        sorted(group)
        for group in registration_result.get("assembly_groups", [])
        if group
    ]
    merged_assemblies = bool(
        registration_result.get("merged_assemblies", len(assembly_groups) <= 1)
    )
    placed_fragments = sorted(transformations.keys())
    for index, group in enumerate(assembly_groups, start=1):
        LOGGER.info("Assembly %d: %d fragment(s) placed", index, len(group))
    LOGGER.info(
        "Placed %d fragment(s) across %d assembly group(s)",
        len(placed_fragments),
        max(len(assembly_groups), 1),
    )

    if pair_alignment_rmse:
        LOGGER.info(
            "Mean pairwise alignment RMSE: %.4f",
            float(np.mean(list(pair_alignment_rmse.values()))),
        )

    _finish_stage("Stage 5: Alignment", stage_start, stage_times)

    # Stage 6 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 6: Reconstruction & Evaluation")
    reconstruction_output = output_dir / "reconstruction.ply"
    if args.fast:
        LOGGER.info(
            "Fast mode: loading only the %d placed full-resolution fragment meshes for final output",
            len(placed_fragments),
        )
        output_meshes = _prepare_stage6_meshes(fragment_metadata, placed_fragments)
    else:
        output_meshes = {
            name: meshes_full[name]
            for name in placed_fragments
            if name in meshes_full
        }
        missing_for_export = [name for name in placed_fragments if name not in output_meshes]
        if missing_for_export:
            output_meshes.update(_prepare_stage6_meshes(fragment_metadata, missing_for_export))

    output_transformations = (
        _offset_assembly_groups(output_meshes, transformations, assembly_groups)
        if len(assembly_groups) > 1 and not merged_assemblies
        else transformations
    )
    assembled_mesh, reconstruction_metrics = assemble_reconstruction(
        output_meshes,
        output_transformations,
        output_dir=output_dir,
        output_name="assembled_reconstruction.ply",
    )

    if len(assembly_groups) > 1 and not merged_assemblies:
        LOGGER.info(
            "Keeping %d disconnected assemblies as separate reconstruction groups",
            len(assembly_groups),
        )
        for index, group in enumerate(assembly_groups, start=1):
            group_meshes = {
                name: output_meshes[name]
                for name in group
                if name in output_meshes
            }
            group_transforms = {
                name: output_transformations[name]
                for name in group
                if name in output_transformations
            }
            assemble_reconstruction(
                group_meshes,
                group_transforms,
                output_dir=output_dir,
                output_name=f"assembled_reconstruction_group_{index}.ply",
                compute_metrics=False,
            )

    assembled_bbox = assembled_mesh.get_axis_aligned_bounding_box()
    bbox_extent = assembled_bbox.get_extent()
    bbox_volume = float(np.prod(np.maximum(bbox_extent, EPSILON))) if len(bbox_extent) else 0.0
    assembled_points = _sample_reconstruction_points(
        output_meshes,
        output_transformations,
        total_points=DEFAULT_PLOTLY_POINTS,
    )
    assembled_output = output_dir / "assembled_reconstruction.ply"
    if _paths_alias_same_file(assembled_output, reconstruction_output):
        LOGGER.info(
            "Skipping reconstruction copy because %s already aliases %s",
            reconstruction_output,
            assembled_output,
        )
    else:
        copyfile(assembled_output, reconstruction_output)

    gap_volume_estimate = reconstruction_metrics.get("total_gap_volume_estimate")
    estimated_gap_percentage = (
        None
        if gap_volume_estimate is None
        else 100.0 * float(gap_volume_estimate) / max(bbox_volume, EPSILON)
    )

    evaluation_metrics = {
        "num_fragments_loaded": len(fragment_names),
        "num_fragments_successfully_placed": len(placed_fragments),
        "num_assemblies": len(assembly_groups) if assembly_groups else 1,
        "mean_alignment_rmse": float(np.mean(list(pair_alignment_rmse.values()))) if pair_alignment_rmse else None,
        "estimated_gap_percentage": None if estimated_gap_percentage is None else float(estimated_gap_percentage),
        "surface_classification_accuracy": (
            None if args.skip_classification else getattr(surface_model, "validation_accuracy_", None)
        ),
    }
    # TODO: Replace this pseudo-label holdout score with a real manually labeled
    # validation metric once annotations are available.

    LOGGER.info("Fragments successfully placed: %d / %d", len(placed_fragments), len(fragment_names))
    LOGGER.info("Estimated gap percentage: %.2f%%", evaluation_metrics["estimated_gap_percentage"])

    plot_metrics_summary(
        {
            "pair_alignment_rmse": pair_alignment_rmse,
            "gap_distances": gap_distances,
            "surface_breakdown": surface_breakdown,
        },
        output_dir=output_dir,
    )

    if len(assembled_points):
        export_reconstruction_plotly(assembled_points, output_dir=output_dir)

    if args.visualize:
        transformed_fragments = _transform_fragments(
            output_meshes,
            output_transformations,
        )
        visualize_reconstruction(
            assembled_mesh,
            individual_fragments=transformed_fragments,
            output_dir=output_dir,
        )

    _finish_stage("Stage 6: Reconstruction & Evaluation", stage_start, stage_times)

    # Stage 7 -----------------------------------------------------------------
    stage_start = _start_stage("Stage 7: Summary Report")
    failed_fragments = sorted(set(fragment_names) - set(placed_fragments))
    total_runtime = time.perf_counter() - total_start

    print("\nFinal Summary")
    print("-" * 72)
    summary_rows = [
        ("Fragments loaded", len(fragment_names)),
        *[
            (f"Assembly {index}", f"{len(group)} fragments placed")
            for index, group in enumerate(assembly_groups, start=1)
        ],
        ("Total", f"{len(placed_fragments)} / {len(fragment_names)} fragments placed"),
        ("Mean alignment RMSE", evaluation_metrics["mean_alignment_rmse"]),
        ("Estimated gap percentage", f'{evaluation_metrics["estimated_gap_percentage"]:.2f}%'),
        ("Surface classification accuracy", evaluation_metrics["surface_classification_accuracy"]),
        ("Total runtime (s)", total_runtime),
    ]
    for label, value in summary_rows:
        print(f"{label:<32} {_format_metric(value)}")

    print("\nPlaced Fragments")
    print("-" * 72)
    print(", ".join(placed_fragments) if placed_fragments else "None")

    print("\nFragments With Issues")
    print("-" * 72)
    if fragment_issues:
        for name in sorted(fragment_issues):
            if fragment_issues[name]:
                print(f"{name}: {' | '.join(fragment_issues[name])}")
    else:
        print("None")

    _finish_stage("Stage 7: Summary Report", stage_start, stage_times)
    total_runtime = time.perf_counter() - total_start

    metrics_payload = {
        "evaluation_metrics": evaluation_metrics,
        "reconstruction_metrics": reconstruction_metrics,
        "refined_matches": refined_matches,
        "pair_alignment_rmse": pair_alignment_rmse,
        "surface_breakdown": surface_breakdown,
        "classification_summary": classification_summary,
        "stage_times_seconds": stage_times,
        "total_runtime_seconds": total_runtime,
        "placed_fragments": placed_fragments,
        "assembly_groups": assembly_groups,
        "merged_assemblies": merged_assemblies,
        "fragments_with_issues": {name: issues for name, issues in fragment_issues.items() if issues},
        "unplaced_fragments": failed_fragments,
    }
    _save_json(output_dir / "metrics.json", metrics_payload)

    LOGGER.info("Saved reconstruction to %s", reconstruction_output)
    LOGGER.info("Saved metrics to %s", output_dir / "metrics.json")

    print("\nStage Timings")
    print("-" * 72)
    for stage_name, seconds in stage_times.items():
        print(f"{stage_name:<32} {seconds:>8.2f}s")

    LOGGER.info("Total pipeline runtime: %.2f seconds", total_runtime)


if __name__ == "__main__":
    main()
