"""Pairwise fragment matching utilities."""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.config import LOGGER, MATCH_MAX_POINTS
from src.feature_extraction import compute_basic_descriptors

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None


MatchRecord = Tuple[int, int, float]
ScoredMatch = Dict[str, Any]
ENABLE_EXPERIMENTAL_ML = False
EPSILON = 1e-8
MIN_RANSAC_CORRESPONDENCES = 20


def _get_break_pointcloud(
    break_entry: o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray],
) -> o3d.geometry.PointCloud:
    """Extract a point cloud from a break-surface dictionary entry."""
    if isinstance(break_entry, tuple):
        return break_entry[0]
    return break_entry


def _pointcloud_to_array(pcd: o3d.geometry.PointCloud, max_points: int = 5000) -> np.ndarray:
    """Convert a point cloud to a possibly downsampled numpy array."""
    points = np.asarray(pcd.points)
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def _subset_feature(
    feature: o3d.pipelines.registration.Feature, indices: np.ndarray
) -> o3d.pipelines.registration.Feature:
    """Subset an Open3D feature object so it stays aligned with a sampled cloud."""
    subset = o3d.pipelines.registration.Feature()
    subset.data = np.asarray(feature.data)[:, indices]
    return subset


def _cap_feature_cloud(
    pcd: o3d.geometry.PointCloud,
    feature: o3d.pipelines.registration.Feature,
    max_points: int = MATCH_MAX_POINTS,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Cap the point count before RANSAC to keep matching tractable."""
    if len(pcd.points) <= max_points:
        return pcd, feature

    indices = np.linspace(0, len(pcd.points) - 1, max_points, dtype=int)
    capped_pcd = pcd.select_by_index(indices.tolist())
    capped_feature = _subset_feature(feature, indices)
    return capped_pcd, capped_feature


def _bbox_diagonal(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> float:
    """Estimate a characteristic scale from two point clouds."""
    source_extent = source_pcd.get_axis_aligned_bounding_box().get_extent()
    target_extent = target_pcd.get_axis_aligned_bounding_box().get_extent()
    combined_extent = np.maximum(source_extent, target_extent)
    diagonal = float(np.linalg.norm(combined_extent))
    return max(diagonal, 1e-6)


def _estimate_overlap_ratio(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> float:
    """Estimate symmetric overlap after applying a candidate transformation."""
    source_points = _pointcloud_to_array(source_pcd)
    target_points = _pointcloud_to_array(target_pcd)
    if len(source_points) == 0 or len(target_points) == 0:
        return 0.0

    source_h = np.hstack([source_points, np.ones((len(source_points), 1), dtype=float)])
    aligned_source = (transformation @ source_h.T).T[:, :3]

    threshold = _bbox_diagonal(source_pcd, target_pcd) * 0.05
    threshold = max(threshold, 1e-4)

    target_tree = cKDTree(target_points)
    source_tree = cKDTree(aligned_source)

    forward_distances, _ = target_tree.query(aligned_source, k=1)
    backward_distances, _ = source_tree.query(target_points, k=1)

    forward_overlap = float(np.mean(forward_distances <= threshold))
    backward_overlap = float(np.mean(backward_distances <= threshold))
    return float(np.clip(0.5 * (forward_overlap + backward_overlap), 0.0, 1.0))


def _safe_fitness(reg_result: o3d.pipelines.registration.RegistrationResult) -> float:
    """Extract a bounded fitness score from an Open3D registration result."""
    return float(np.clip(getattr(reg_result, "fitness", 0.0), 0.0, 1.0))


def _correspondence_count(
    reg_result: o3d.pipelines.registration.RegistrationResult,
) -> int:
    """Read the inlier correspondence count from a registration result."""
    correspondence_set = getattr(reg_result, "correspondence_set", [])
    return int(len(correspondence_set))


def match_fpfh(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float = 0.5,
    source_voxel_size: float | None = None,
    target_voxel_size: float | None = None,
) -> o3d.pipelines.registration.RegistrationResult:
    """Match two break-surface point clouds using FPFH + RANSAC."""
    if source_pcd.is_empty() or target_pcd.is_empty():
        raise ValueError("Source and target point clouds must be non-empty")

    source_scale = float(source_voxel_size if source_voxel_size is not None else voxel_size)
    target_scale = float(target_voxel_size if target_voxel_size is not None else voxel_size)
    if source_scale <= 0 or target_scale <= 0:
        raise ValueError("voxel_size must be positive")

    distance_threshold = max(source_scale, target_scale) * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
    )


def compute_match_score(
    reg_result: o3d.pipelines.registration.RegistrationResult,
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
) -> float:
    """Combine fitness, alignment error, and overlap into a single match score."""
    fitness = _safe_fitness(reg_result)
    rmse = float(max(getattr(reg_result, "inlier_rmse", np.inf), 0.0))
    overlap_ratio = _estimate_overlap_ratio(
        source_pcd,
        target_pcd,
        reg_result.transformation,
    )

    scale = _bbox_diagonal(source_pcd, target_pcd) * 0.05
    scale = max(scale, 1e-4)
    rmse_score = float(np.exp(-rmse / scale)) if np.isfinite(rmse) else 0.0

    score = 0.45 * fitness + 0.25 * rmse_score + 0.30 * overlap_ratio
    # TODO: Learn these weights from validation pairs instead of hard-coding them.
    return float(np.clip(score, 0.0, 1.0))


def _build_score_matrix(
    names: List[str], pair_scores: Dict[Tuple[str, str], float]
) -> np.ndarray:
    """Build a symmetric pairwise score matrix for printing."""
    matrix = np.full((len(names), len(names)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)

    for (name_a, name_b), score in pair_scores.items():
        i = names.index(name_a)
        j = names.index(name_b)
        matrix[i, j] = score
        matrix[j, i] = score
    return matrix


def _print_match_matrix(names: List[str], pair_scores: Dict[Tuple[str, str], float]) -> None:
    """Print a compact matrix of pairwise fragment scores."""
    matrix = _build_score_matrix(names, pair_scores)
    short_names = [_short_fragment_name(name) for name in names]
    header = "fragment".ljust(14) + "".join(name.rjust(12) for name in short_names)
    print("\nPairwise match matrix")
    print(header)

    for row_name, row_values in zip(short_names, matrix):
        cells = []
        for value in row_values:
            if np.isnan(value):
                cells.append("    --     ")
            else:
                cells.append(f"{value:12.3f}")
        print(row_name.ljust(14) + "".join(cells))


def _short_fragment_name(name: str) -> str:
    """Generate a compact, readable fragment label for console matrices."""
    match = re.search(r"FR_(\d+)", name)
    if match:
        return f"FR_{int(match.group(1)):02d}"
    return name[:10]


def find_all_matches(
    fragments_dict: Dict[str, object],
    break_surfaces_dict: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ],
    features_dict: Dict[str, Dict[str, object]],
    score_threshold: float = 0.3,
) -> List[ScoredMatch]:
    """Evaluate all fragment pairs and return plausible neighbors."""
    fragment_names = sorted(set(fragments_dict) & set(break_surfaces_dict) & set(features_dict))
    matches: List[ScoredMatch] = []
    pair_scores: Dict[Tuple[str, str], float] = {}
    pair_names = list(combinations(fragment_names, 2))

    for name_a, name_b in tqdm(
        pair_names,
        desc="Pairwise matching",
        leave=False,
        disable=len(pair_names) < 4,
    ):
        source_pcd = _get_break_pointcloud(break_surfaces_dict[name_a])
        target_pcd = _get_break_pointcloud(break_surfaces_dict[name_b])
        source_features = features_dict[name_a]
        target_features = features_dict[name_b]

        if source_pcd.is_empty() or target_pcd.is_empty():
            LOGGER.warning("Skipping match (%s, %s) because a break surface is empty", name_a, name_b)
            continue

        try:
            source_match_pcd, source_match_fpfh = _cap_feature_cloud(
                source_features["downsampled_pcd"],
                source_features["fpfh"],
            )
            target_match_pcd, target_match_fpfh = _cap_feature_cloud(
                target_features["downsampled_pcd"],
                target_features["fpfh"],
            )
            source_voxel_size = float(source_features.get("fpfh_voxel_size", 0.5))
            target_voxel_size = float(target_features.get("fpfh_voxel_size", 0.5))
            source_break_count = int(source_features.get("break_point_count", len(source_pcd.points)))
            target_break_count = int(target_features.get("break_point_count", len(target_pcd.points)))
            is_large_large_pair = (
                source_break_count > 5000
                and target_break_count > 5000
            )

            scale_factors = [0.5, 1.0, 2.0] if is_large_large_pair else [1.0]
            scale_results: List[tuple[float, o3d.pipelines.registration.RegistrationResult, int, float]] = []
            for scale_factor in scale_factors:
                scaled_source_voxel = max(source_voxel_size * scale_factor, 1e-3)
                scaled_target_voxel = max(target_voxel_size * scale_factor, 1e-3)
                candidate_reg_result = match_fpfh(
                    source_match_pcd,
                    target_match_pcd,
                    source_match_fpfh,
                    target_match_fpfh,
                    voxel_size=max(scaled_source_voxel, scaled_target_voxel),
                    source_voxel_size=scaled_source_voxel,
                    target_voxel_size=scaled_target_voxel,
                )
                candidate_correspondence_count = _correspondence_count(candidate_reg_result)
                if candidate_correspondence_count < MIN_RANSAC_CORRESPONDENCES:
                    continue

                candidate_score = compute_match_score(
                    candidate_reg_result,
                    source_match_pcd,
                    target_match_pcd,
                )
                scale_results.append(
                    (
                        float(scale_factor),
                        candidate_reg_result,
                        int(candidate_correspondence_count),
                        float(candidate_score),
                    )
                )

            if not scale_results:
                pair_scores[(name_a, name_b)] = 0.0
                continue

            scale_results.sort(key=lambda item: item[3], reverse=True)
            best_scale_factor, reg_result, correspondence_count, score = scale_results[0]
            if is_large_large_pair and best_scale_factor != 1.0:
                baseline_scores = [item for item in scale_results if abs(item[0] - 1.0) < EPSILON]
                baseline_score = baseline_scores[0][3] if baseline_scores else 0.0
                LOGGER.info(
                    "Multi-scale RANSAC improved %s vs %s at scale %.2f (score %.3f -> %.3f)",
                    _short_fragment_name(name_a),
                    _short_fragment_name(name_b),
                    best_scale_factor,
                    float(baseline_score),
                    float(score),
                )

            ransac_fitness = _safe_fitness(reg_result)
            if ransac_fitness < 0.05:
                # Very weak RANSAC fits are not worth spending later ICP time on.
                pair_scores[(name_a, name_b)] = float(ransac_fitness)
                continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Matching failed for (%s, %s): %s", name_a, name_b, exc)
            score = 0.0
            reg_result = None
            correspondence_count = 0
            best_scale_factor = 1.0

        pair_scores[(name_a, name_b)] = score
        if reg_result is not None and score > 0.0 and score >= score_threshold:
            matches.append(
                {
                    "fragment_a": name_a,
                    "fragment_b": name_b,
                    "score": float(score),
                    "transformation": reg_result.transformation.copy(),
                    "ransac_fitness": float(_safe_fitness(reg_result)),
                    "ransac_inlier_rmse": float(getattr(reg_result, "inlier_rmse", 0.0)),
                    "correspondence_count": int(_correspondence_count(reg_result)),
                    "source_voxel_size": float(source_voxel_size),
                    "target_voxel_size": float(target_voxel_size),
                    "match_scale_factor": float(best_scale_factor),
                }
            )

    _print_match_matrix(fragment_names, pair_scores)
    return sorted(matches, key=lambda item: float(item["score"]), reverse=True)


def filter_matches(
    matches: List[ScoredMatch], max_matches_per_fragment: int = 3
) -> List[ScoredMatch]:
    """Keep only the strongest and most plausible neighbors for each fragment."""
    if max_matches_per_fragment <= 0:
        raise ValueError("max_matches_per_fragment must be positive")

    if not matches:
        return []

    best_score_by_fragment: Dict[str, float] = {}
    for match in matches:
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        score = float(match["score"])
        best_score_by_fragment[name_a] = max(best_score_by_fragment.get(name_a, 0.0), score)
        best_score_by_fragment[name_b] = max(best_score_by_fragment.get(name_b, 0.0), score)

    filtered_candidates = [
        match
        for match in matches
        if float(match["score"])
        >= 0.5
        * best_score_by_fragment[str(match["fragment_a"])]  # 50% felt like a decent cutoff here
        and float(match["score"])
        >= 0.5 * best_score_by_fragment[str(match["fragment_b"])]
    ]

    filtered_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    kept: List[ScoredMatch] = []
    match_counts: Dict[str, int] = {}

    for match in filtered_candidates:
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        if match_counts.get(name_a, 0) >= max_matches_per_fragment:
            continue
        if match_counts.get(name_b, 0) >= max_matches_per_fragment:
            continue

        kept.append(match)
        match_counts[name_a] = match_counts.get(name_a, 0) + 1
        match_counts[name_b] = match_counts.get(name_b, 0) + 1

    return kept


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    numerator = float(np.dot(vector_a, vector_b))
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + EPSILON)
    similarity = numerator / denominator
    return float(np.clip(similarity, -1.0, 1.0))


def _estimate_break_surface_area(pcd: o3d.geometry.PointCloud) -> float:
    """Estimate point-cloud patch area from density and point count."""
    points = np.asarray(pcd.points)
    if len(points) < 3:
        return 0.0

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    if distances.ndim == 1:
        mean_spacing = float(np.mean(distances))
    else:
        mean_spacing = float(np.mean(distances[:, -1]))
    return max(len(points) * mean_spacing * mean_spacing, EPSILON)


def _profile_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Compare two surface profiles using normalized cross-correlation."""
    heights_a = np.asarray(profile_a)[:, 1]
    heights_b = np.asarray(profile_b)[:, 1]

    target_length = min(len(heights_a), len(heights_b))
    if target_length == 0:
        return 0.0

    x_a = np.linspace(0.0, 1.0, len(heights_a))
    x_b = np.linspace(0.0, 1.0, len(heights_b))
    x_target = np.linspace(0.0, 1.0, target_length)
    series_a = np.interp(x_target, x_a, heights_a)
    series_b = np.interp(x_target, x_b, heights_b)

    series_a = series_a - series_a.mean()
    series_b = series_b - series_b.mean()
    norm_a = np.linalg.norm(series_a)
    norm_b = np.linalg.norm(series_b)
    if norm_a < EPSILON or norm_b < EPSILON:
        return 0.0

    correlation = np.correlate(series_a / norm_a, series_b / norm_b, mode="full")
    max_corr = float(np.max(correlation) / target_length)
    return float(np.clip((max_corr + 1.0) * 0.5, 0.0, 1.0))


def _fpfh_distance_stats(
    fpfh_a: np.ndarray, fpfh_b: np.ndarray, max_rows: int = 512
) -> Tuple[float, float]:
    """Compute mean and std of nearest-neighbor FPFH descriptor distances."""
    if len(fpfh_a) == 0 or len(fpfh_b) == 0:
        return 0.0, 0.0

    if len(fpfh_a) > max_rows:
        indices_a = np.linspace(0, len(fpfh_a) - 1, max_rows, dtype=int)
        fpfh_a = fpfh_a[indices_a]
    if len(fpfh_b) > max_rows:
        indices_b = np.linspace(0, len(fpfh_b) - 1, max_rows, dtype=int)
        fpfh_b = fpfh_b[indices_b]

    distances = cdist(fpfh_a, fpfh_b, metric="euclidean")
    nearest = distances.min(axis=1)
    return float(nearest.mean()), float(nearest.std())


def _build_pair_features(
    name_a: str,
    name_b: str,
    break_surfaces_dict: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ],
    features_dict: Dict[str, Dict[str, object]],
) -> np.ndarray:
    """Assemble a compact learned feature vector for one fragment pair."""
    break_a = _get_break_pointcloud(break_surfaces_dict[name_a])
    break_b = _get_break_pointcloud(break_surfaces_dict[name_b])
    features_a = features_dict[name_a]
    features_b = features_dict[name_b]

    histogram_similarity = _cosine_similarity(
        np.asarray(features_a["curvature_histogram"], dtype=float),
        np.asarray(features_b["curvature_histogram"], dtype=float),
    )

    area_a = _estimate_break_surface_area(break_a)
    area_b = _estimate_break_surface_area(break_b)
    area_ratio = min(area_a, area_b) / max(area_a, area_b, EPSILON)

    profile_similarity = _profile_similarity(
        np.asarray(features_a["surface_profile"], dtype=float),
        np.asarray(features_b["surface_profile"], dtype=float),
    )

    mean_distance, std_distance = _fpfh_distance_stats(
        np.asarray(features_a["fpfh_matrix"], dtype=float),
        np.asarray(features_b["fpfh_matrix"], dtype=float),
    )

    return np.array(
        [
            histogram_similarity,
            area_ratio,
            profile_similarity,
            mean_distance,
            std_distance,
        ],
        dtype=np.float32,
    )


def predict_adjacency_ml(
    fragments_dict: Dict[str, object],
    break_surfaces_dict: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ],
    features_dict: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    """Predict fragment adjacency with a small experimental MLP."""
    if not ENABLE_EXPERIMENTAL_ML:
        LOGGER.info(
            "Experimental ML adjacency prediction is disabled. "
            "Set ENABLE_EXPERIMENTAL_ML = True in matching.py to enable it."
        )
        return []

    if torch is None or nn is None:
        raise ImportError("torch is required for experimental adjacency prediction")

    fragment_names = sorted(set(fragments_dict) & set(break_surfaces_dict) & set(features_dict))
    if len(fragment_names) < 2:
        return []

    pair_names: List[Tuple[str, str]] = []
    pair_features: List[np.ndarray] = []
    soft_targets: List[float] = []

    for name_a, name_b in combinations(fragment_names, 2):
        features = _build_pair_features(name_a, name_b, break_surfaces_dict, features_dict)
        pair_names.append((name_a, name_b))
        pair_features.append(features)

        try:
            source_match_pcd, source_match_fpfh = _cap_feature_cloud(
                features_dict[name_a]["downsampled_pcd"],
                features_dict[name_a]["fpfh"],
            )
            target_match_pcd, target_match_fpfh = _cap_feature_cloud(
                features_dict[name_b]["downsampled_pcd"],
                features_dict[name_b]["fpfh"],
            )
            reg_result = match_fpfh(
                source_match_pcd,
                target_match_pcd,
                source_match_fpfh,
                target_match_fpfh,
                voxel_size=max(
                    float(features_dict[name_a].get("fpfh_voxel_size", 0.5)),
                    float(features_dict[name_b].get("fpfh_voxel_size", 0.5)),
                ),
            )
            soft_targets.append(_safe_fitness(reg_result))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Experimental ML supervision failed for (%s, %s): %s", name_a, name_b, exc)
            soft_targets.append(0.0)

    x = np.vstack(pair_features).astype(np.float32)
    y = np.asarray(soft_targets, dtype=np.float32).reshape(-1, 1)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    # This is a lightweight learned approach that supplements RANSAC matching.
    # With labeled training data this could be much more powerful.
    model = nn.Sequential(
        nn.Linear(x.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_scores = model(x_tensor).squeeze(1).cpu().numpy()

    predictions: List[Dict[str, object]] = []
    for (name_a, name_b), score, target in zip(pair_names, predicted_scores, soft_targets):
        confidence = float(np.clip(abs(score - 0.5) * 2.0, 0.0, 1.0))
        predictions.append(
            {
                "fragment_a": name_a,
                "fragment_b": name_b,
                "adjacency_score": float(score),
                "confidence": confidence,
                "soft_target": float(target),
                "enabled": ENABLE_EXPERIMENTAL_ML,
            }
        )

    predictions.sort(key=lambda item: item["adjacency_score"], reverse=True)
    return predictions


def descriptor_distance(
    mesh_a: o3d.geometry.TriangleMesh, mesh_b: o3d.geometry.TriangleMesh
) -> float:
    """Compare two meshes with a simple descriptor distance."""
    desc_a = compute_basic_descriptors(mesh_a)
    desc_b = compute_basic_descriptors(mesh_b)

    vec_a = np.array(list(desc_a.values()), dtype=float)
    vec_b = np.array(list(desc_b.values()), dtype=float)
    return float(np.linalg.norm(vec_a - vec_b))


def score_fragment_pairs(meshes: List[o3d.geometry.TriangleMesh]) -> List[MatchRecord]:
    """Compatibility helper that scores mesh pairs with simple global descriptors."""
    matches: List[MatchRecord] = []
    for i, j in combinations(range(len(meshes)), 2):
        score = descriptor_distance(meshes[i], meshes[j])
        matches.append((i, j, score))
    return sorted(matches, key=lambda item: item[2])


def summarize_matches(matches: Iterable[Tuple[int, int, float]], top_k: int = 10) -> List[Dict[str, float]]:
    """Convert top matches into a serializable summary."""
    summary = []
    for idx_a, idx_b, score in list(matches)[:top_k]:
        summary.append({"fragment_a": idx_a, "fragment_b": idx_b, "score": score})
    return summary
