"""Registration and alignment helpers.

Gap handling is one of the hardest parts of this project. Fragments separated
by material loss will not have perfect ICP convergence, so we use generous
thresholds and rely on the RANSAC initialization to get us into the right basin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from src.config import ICP_THRESHOLD, LOGGER, OUTPUT_DIR, RANSAC_DISTANCE
from src.feature_extraction import compute_fpfh, mesh_to_point_cloud


EPSILON = 1e-8
GeometryType = o3d.geometry.TriangleMesh | o3d.geometry.PointCloud
PAIR_REFINED_FITNESS_THRESHOLD = 0.05
SEED_FITNESS_THRESHOLD = 0.10
ASSEMBLY_EXTENSION_THRESHOLD = 0.08
SECOND_PASS_FITNESS_THRESHOLD = 0.12
MERGE_FITNESS_THRESHOLD = 0.08
SECOND_PASS_BREAK_POINT_THRESHOLD = 1000
MIN_RANSAC_CORRESPONDENCES = 20


def _copy_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Return a shallow copy of a mesh suitable for transformations."""
    return o3d.geometry.TriangleMesh(mesh)


def _copy_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Return a copy of a point cloud."""
    return o3d.geometry.PointCloud(pcd)


def _copy_geometry(geometry: GeometryType) -> GeometryType:
    """Copy either a mesh or a point cloud."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        return _copy_mesh(geometry)
    if isinstance(geometry, o3d.geometry.PointCloud):
        return _copy_pointcloud(geometry)
    raise TypeError("Unsupported geometry type")


def _ensure_normals(
    pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30
) -> o3d.geometry.PointCloud:
    """Ensure a point cloud has normals for point-to-plane ICP."""
    if pcd.is_empty():
        raise ValueError("Point cloud is empty")

    working_pcd = _copy_pointcloud(pcd)
    if not working_pcd.has_normals() or len(working_pcd.normals) != len(working_pcd.points):
        working_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(radius, 1e-3), max_nn=max_nn)
        )
        working_pcd.normalize_normals()
    return working_pcd


def _downsample_for_icp(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    """Voxel-downsample a point cloud while keeping a usable fallback."""
    if voxel_size <= 0:
        return _copy_pointcloud(pcd)

    downsampled = pcd.voxel_down_sample(voxel_size)
    if downsampled.is_empty():
        return _copy_pointcloud(pcd)
    return downsampled


def _pointcloud_from_mesh(
    mesh: o3d.geometry.TriangleMesh, num_points: int = 10000
) -> o3d.geometry.PointCloud:
    """Sample a point cloud from a mesh and ensure normals exist."""
    pcd = mesh_to_point_cloud(mesh, num_points=num_points)
    bbox_extent = np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())
    radius = max(bbox_extent * 0.05, 1e-3)
    return _ensure_normals(pcd, radius=radius)


def _pointcloud_from_geometry(
    geometry: GeometryType,
    num_points: int = 10000,
) -> o3d.geometry.PointCloud:
    """Convert either a mesh or point cloud into an ICP-ready point cloud."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        return _pointcloud_from_mesh(geometry, num_points=num_points)
    if isinstance(geometry, o3d.geometry.PointCloud):
        bbox_extent = np.linalg.norm(geometry.get_axis_aligned_bounding_box().get_extent())
        radius = max(bbox_extent * 0.05, 1e-3)
        return _ensure_normals(geometry, radius=radius)
    raise TypeError("Unsupported geometry type")


def _geometry_diagonal(geometry: GeometryType) -> float:
    """Measure fragment size from the full working geometry bounding box."""
    extent = geometry.get_axis_aligned_bounding_box().get_extent()
    diagonal = float(np.linalg.norm(extent))
    return max(diagonal, 1e-6)


def _transform_geometry(geometry: GeometryType, transformation: np.ndarray) -> GeometryType:
    """Apply a rigid transform to either a mesh or point cloud copy."""
    transformed = _copy_geometry(geometry)
    transformed.transform(transformation)

    if isinstance(transformed, o3d.geometry.TriangleMesh):
        transformed.compute_vertex_normals()
    elif isinstance(transformed, o3d.geometry.PointCloud) and transformed.has_normals():
        transformed.normalize_normals()
    return transformed


def _merge_pointclouds(pointclouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds into one."""
    merged = o3d.geometry.PointCloud()
    if not pointclouds:
        return merged

    points = []
    normals = []
    has_all_normals = all(pcd.has_normals() and len(pcd.normals) == len(pcd.points) for pcd in pointclouds)

    for pcd in pointclouds:
        if pcd.is_empty():
            continue
        points.append(np.asarray(pcd.points))
        if has_all_normals:
            normals.append(np.asarray(pcd.normals))

    if not points:
        return merged

    merged.points = o3d.utility.Vector3dVector(np.vstack(points))
    if has_all_normals and normals:
        merged.normals = o3d.utility.Vector3dVector(np.vstack(normals))
    return merged


def _pair_distance_stats(
    source_geometry: GeometryType,
    target_geometry: GeometryType,
    max_points: int = 5000,
) -> Dict[str, float]:
    """Estimate gap statistics between two aligned meshes."""
    source_pcd = _pointcloud_from_geometry(source_geometry, num_points=max_points)
    target_pcd = _pointcloud_from_geometry(target_geometry, num_points=max_points)

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    if len(source_points) == 0 or len(target_points) == 0:
        return {"min_distance": 0.0, "mean_distance": 0.0, "max_gap": 0.0, "rmse": 0.0}

    target_tree = cKDTree(target_points)
    source_tree = cKDTree(source_points)
    forward, _ = target_tree.query(source_points, k=1)
    backward, _ = source_tree.query(target_points, k=1)
    distances = np.concatenate([forward, backward])

    return {
        "min_distance": float(distances.min()),
        "mean_distance": float(distances.mean()),
        "max_gap": float(distances.max()),
        "rmse": float(np.sqrt(np.mean(distances**2))),
    }


def _compose_global_transform(
    placed_name: str,
    new_name: str,
    pair_match: Dict[str, Any],
    placed_transforms: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute a new fragment transform from an already placed neighbor."""
    name_a = str(pair_match["fragment_a"])
    name_b = str(pair_match["fragment_b"])
    pair_transform = np.asarray(pair_match["transformation"], dtype=float)
    if placed_name == name_a and new_name == name_b:
        return placed_transforms[placed_name] @ np.linalg.inv(pair_transform)
    if placed_name == name_b and new_name == name_a:
        return placed_transforms[placed_name] @ pair_transform
    raise ValueError("Pair match does not connect the requested fragments")


def _get_break_pointcloud(
    break_entry: o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray] | None,
) -> o3d.geometry.PointCloud | None:
    """Unpack break-surface entries that may carry indices alongside the points."""
    if break_entry is None:
        return None
    if isinstance(break_entry, tuple):
        return break_entry[0]
    return break_entry


def _short_fragment_name(name: str) -> str:
    """Make long dataset filenames easier to read in logs."""
    parts = name.split("FR_")
    if len(parts) > 1:
        suffix = parts[1][:2]
        if suffix.isdigit():
            return f"FR_{int(suffix):02d}"
    return name


def _size_score(
    match: Dict[str, Any],
    fragment_diagonals: Dict[str, float],
    max_diagonal: float,
) -> float:
    """Normalize the physical size of a fragment pair to the [0, 1] range."""
    name_a = str(match["fragment_a"])
    name_b = str(match["fragment_b"])
    pair_diagonal = fragment_diagonals.get(name_a, 0.0) + fragment_diagonals.get(name_b, 0.0)
    normalizer = max(max_diagonal * 2.0, EPSILON)
    return float(np.clip(pair_diagonal / normalizer, 0.0, 1.0))


def _combined_seed_score(
    match: Dict[str, Any],
    fragment_diagonals: Dict[str, float],
    max_diagonal: float,
) -> float:
    """Balance RANSAC, refined ICP, and fragment importance when choosing a seed."""
    size_score = _size_score(match, fragment_diagonals, max_diagonal)
    return float(
        0.4 * float(match.get("score", 0.0))
        + 0.4 * float(match.get("refined_fitness", 0.0))
        + 0.2 * size_score
    )


def _build_connected_components(
    fragment_names: List[str],
    matches: List[Dict[str, Any]],
) -> List[List[str]]:
    """Find connected components in the accepted pairwise match graph."""
    adjacency: Dict[str, set[str]] = {name: set() for name in fragment_names}
    for match in matches:
        name_a = str(match["fragment_a"])
        name_b = str(match["fragment_b"])
        adjacency.setdefault(name_a, set()).add(name_b)
        adjacency.setdefault(name_b, set()).add(name_a)

    visited: set[str] = set()
    components: List[List[str]] = []
    for name in fragment_names:
        if name in visited:
            continue

        stack = [name]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited))
        components.append(sorted(component))
    return components


def _assembly_pointcloud(
    fragments: Dict[str, GeometryType],
    transformations: Dict[str, np.ndarray],
    names: List[str],
) -> o3d.geometry.PointCloud:
    """Merge a subset of transformed fragments into one point cloud."""
    pointclouds: List[o3d.geometry.PointCloud] = []
    for name in names:
        if name not in transformations:
            continue
        geometry_world = _transform_geometry(fragments[name], transformations[name])
        pointclouds.append(_pointcloud_from_geometry(geometry_world, num_points=8000))
    return _merge_pointclouds(pointclouds)


def _refine_world_alignment(
    source_geometry: GeometryType,
    target_geometry: GeometryType,
    initial_transform: np.ndarray,
    voxel_sizes: List[float],
    acceptance_threshold: float,
) -> Dict[str, object]:
    """Refine a proposed world-frame placement and flag weak alignments."""
    source_world = _transform_geometry(source_geometry, initial_transform)
    source_world_pcd = _pointcloud_from_geometry(source_world, num_points=12000)
    target_world_pcd = _pointcloud_from_geometry(target_geometry, num_points=12000)

    refined_world = multi_scale_icp(
        source_world_pcd,
        target_world_pcd,
        initial_transform=np.eye(4),
        voxel_sizes=voxel_sizes,
        rejection_threshold=acceptance_threshold,
    )
    refined_delta = np.asarray(refined_world["transformation"], dtype=float)
    final_transform = refined_delta @ initial_transform
    return {
        "transformation": final_transform,
        "fitness": float(refined_world["fitness"]),
        "inlier_rmse": float(refined_world["inlier_rmse"]),
        "correspondence_count": int(refined_world["correspondence_count"]),
        "rejected": bool(refined_world.get("rejected", False)),
    }


def _aggregate_placed_targets(
    fragments: Dict[str, GeometryType],
    transformations: Dict[str, np.ndarray],
    exclude_name: str | None = None,
) -> o3d.geometry.PointCloud:
    """Sample all placed fragments into one target point cloud."""
    pointclouds: List[o3d.geometry.PointCloud] = []
    for name, transform in transformations.items():
        if name == exclude_name:
            continue

        geometry_world = _transform_geometry(fragments[name], transform)
        pointclouds.append(_pointcloud_from_geometry(geometry_world, num_points=8000))
    return _merge_pointclouds(pointclouds)


def _estimate_gap_volume_and_rmse(
    transformed_meshes: Dict[str, o3d.geometry.TriangleMesh]
) -> Tuple[float, float]:
    """Estimate a rough total gap volume and alignment RMSE from pairwise distances."""
    names = list(transformed_meshes.keys())
    if len(names) < 2:
        return 0.0, 0.0

    gap_volume = 0.0
    all_squared_distances: List[np.ndarray] = []
    sampled_points: Dict[str, np.ndarray] = {}

    for name in names:
        vertices = np.asarray(transformed_meshes[name].vertices)
        if len(vertices) == 0:
            sampled_points[name] = np.empty((0, 3), dtype=float)
            continue
        if len(vertices) > 3000:
            indices = np.linspace(0, len(vertices) - 1, 3000, dtype=int)
            sampled_points[name] = vertices[indices]
        else:
            sampled_points[name] = vertices

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mesh_a = transformed_meshes[names[i]]
            mesh_b = transformed_meshes[names[j]]
            points_a = sampled_points.get(names[i], np.empty((0, 3), dtype=float))
            points_b = sampled_points.get(names[j], np.empty((0, 3), dtype=float))
            if len(points_a) == 0 or len(points_b) == 0:
                continue

            bbox_extent = np.maximum(
                mesh_a.get_axis_aligned_bounding_box().get_extent(),
                mesh_b.get_axis_aligned_bounding_box().get_extent(),
            )
            contact_threshold = max(float(np.linalg.norm(bbox_extent)) * 0.03, 1e-3)

            tree_b = cKDTree(points_b)
            distances, _ = tree_b.query(points_a, k=1)
            close_distances = distances[distances <= contact_threshold]
            if len(close_distances) == 0:
                continue

            spacing = max(contact_threshold * 0.5, 1e-4)
            contact_area_estimate = len(close_distances) * spacing * spacing
            gap_volume += float(close_distances.mean() * contact_area_estimate)
            all_squared_distances.append(close_distances**2)

    if not all_squared_distances:
        return gap_volume, 0.0

    squared = np.concatenate(all_squared_distances)
    return gap_volume, float(np.sqrt(np.mean(squared)))


def refine_alignment_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    threshold: float = 1.0,
    max_iteration: int = 200,
) -> Dict[str, object]:
    """Refine a candidate alignment with point-to-plane ICP."""
    if source_pcd.is_empty() or target_pcd.is_empty():
        raise ValueError("Source and target point clouds must be non-empty")

    source_diagonal = _geometry_diagonal(source_pcd)
    icp_threshold = max(float(threshold), 1.0, source_diagonal * 0.02)
    source_ready = _ensure_normals(source_pcd, radius=max(icp_threshold * 2.0, 1e-3))
    target_ready = _ensure_normals(target_pcd, radius=max(icp_threshold * 2.0, 1e-3))
    robust_kernel = o3d.pipelines.registration.TukeyLoss(k=icp_threshold)

    result = o3d.pipelines.registration.registration_icp(
        source_ready,
        target_ready,
        icp_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(robust_kernel),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iteration,
        ),
    )

    return {
        "transformation": result.transformation,
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
    }


def multi_scale_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_sizes: List[float] = [2.0, 1.0, 0.5],
    rejection_threshold: float | None = None,
) -> Dict[str, object]:
    """Run ICP from coarse to fine resolution."""
    if source_pcd.is_empty() or target_pcd.is_empty():
        raise ValueError("Source and target point clouds must be non-empty")

    current_transform = np.array(initial_transform, dtype=float, copy=True)
    per_scale_metrics: List[Dict[str, float]] = []

    for voxel_size in voxel_sizes:
        finest_scale = voxel_size == min(voxel_sizes)
        source_scaled = _downsample_for_icp(source_pcd, voxel_size)
        target_scaled = _downsample_for_icp(target_pcd, voxel_size)
        icp_result = refine_alignment_icp(
            source_scaled,
            target_scaled,
            current_transform,
            threshold=max(voxel_size * 2.0, ICP_THRESHOLD),
            max_iteration=500 if finest_scale else 200,
        )
        current_transform = np.array(icp_result["transformation"], dtype=float, copy=True)
        per_scale_metrics.append(
            {
                "voxel_size": float(voxel_size),
                "fitness": float(icp_result["fitness"]),
                "inlier_rmse": float(icp_result["inlier_rmse"]),
                "correspondence_count": float(icp_result["correspondence_count"]),
            }
        )

    final_metrics = per_scale_metrics[-1] if per_scale_metrics else {
        "fitness": 0.0,
        "inlier_rmse": 0.0,
        "correspondence_count": 0.0,
    }

    return {
        "transformation": current_transform,
        "per_scale_metrics": per_scale_metrics,
        "fitness": float(final_metrics["fitness"]),
        "inlier_rmse": float(final_metrics["inlier_rmse"]),
        "correspondence_count": int(final_metrics["correspondence_count"]),
        "rejected": bool(
            rejection_threshold is not None
            and float(final_metrics["fitness"]) < float(rejection_threshold)
        ),
    }


def align_pair(
    source_mesh: GeometryType,
    target_mesh: GeometryType,
    transformation: np.ndarray,
) -> Tuple[GeometryType, Dict[str, float]]:
    """Transform one fragment into another fragment's frame and measure the gap."""
    transformed_geometry = _transform_geometry(source_mesh, transformation)
    gap_stats = _pair_distance_stats(transformed_geometry, target_mesh)
    return transformed_geometry, gap_stats


def run_ransac_alignment(
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
    distance_threshold: float = RANSAC_DISTANCE,
) -> o3d.pipelines.registration.RegistrationResult:
    """Coarse alignment using FPFH + RANSAC."""
    source_pcd = mesh_to_point_cloud(source_mesh)
    target_pcd = mesh_to_point_cloud(target_mesh)
    source_fpfh = compute_fpfh(source_pcd)
    target_fpfh = compute_fpfh(target_pcd)

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
            )
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
    )


def run_icp_refinement(
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
    initial_transform: np.ndarray,
    threshold: float = ICP_THRESHOLD,
) -> o3d.pipelines.registration.RegistrationResult:
    """Compatibility wrapper that returns the raw Open3D ICP result."""
    source_pcd = _pointcloud_from_mesh(source_mesh)
    target_pcd = _pointcloud_from_mesh(target_mesh)
    source_ready = _ensure_normals(source_pcd, radius=max(threshold * 2.0, 1e-3))
    target_ready = _ensure_normals(target_pcd, radius=max(threshold * 2.0, 1e-3))

    return o3d.pipelines.registration.registration_icp(
        source_ready,
        target_ready,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=200,
        ),
    )


def register_pair(
    source_mesh: o3d.geometry.TriangleMesh, target_mesh: o3d.geometry.TriangleMesh
) -> Dict[str, np.ndarray | float]:
    """Run coarse-to-fine alignment for a single fragment pair."""
    ransac_result = run_ransac_alignment(source_mesh, target_mesh)
    source_pcd = _pointcloud_from_mesh(source_mesh)
    target_pcd = _pointcloud_from_mesh(target_mesh)
    icp_result = multi_scale_icp(
        source_pcd,
        target_pcd,
        initial_transform=ransac_result.transformation,
    )

    return {
        "fitness": float(icp_result["fitness"]),
        "inlier_rmse": float(icp_result["inlier_rmse"]),
        "transformation": np.asarray(icp_result["transformation"]),
    }


def _build_component_assembly(
    fragments: Dict[str, GeometryType],
    component_names: List[str],
    component_matches: List[Dict[str, Any]],
    fragment_diagonals: Dict[str, float],
    max_diagonal: float,
    break_surface_sizes: Dict[str, int] | None = None,
    break_surfaces: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ]
    | None = None,
    features_dict: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, Any]:
    """Build a greedy assembly for one connected component of the match graph."""
    component_names = sorted(component_names)
    break_surface_sizes = break_surface_sizes or {}
    if not component_matches:
        return {"transformations": {}, "placed": [], "unplaced": component_names}

    seed_matches = [
        match
        for match in component_matches
        if float(match.get("refined_fitness", 0.0)) >= SEED_FITNESS_THRESHOLD
    ]
    seed_pool = seed_matches if seed_matches else component_matches
    seed_match = max(
        seed_pool,
        key=lambda item: (
            _combined_seed_score(item, fragment_diagonals, max_diagonal),
            float(item.get("refined_fitness", 0.0)),
            float(item.get("score", 0.0)),
        ),
    )

    seed_a = str(seed_match["fragment_a"])
    seed_b = str(seed_match["fragment_b"])
    seed_transform = np.asarray(seed_match["transformation"], dtype=float)
    transformations: Dict[str, np.ndarray] = {
        seed_a: np.eye(4),
        seed_b: np.linalg.inv(seed_transform),
    }
    placed = {seed_a, seed_b}
    remaining = set(component_names) - placed
    ordered_matches = sorted(
        component_matches,
        key=lambda item: (
            float(item.get("refined_fitness", 0.0)),
            _combined_seed_score(item, fragment_diagonals, max_diagonal),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )

    while remaining:
        candidate_matches = [
            match
            for match in ordered_matches
            if (
                str(match["fragment_a"]) in placed
                and str(match["fragment_b"]) in remaining
            )
            or (
                str(match["fragment_b"]) in placed
                and str(match["fragment_a"]) in remaining
            )
        ]

        if not candidate_matches:
            LOGGER.warning(
                "Could not connect remaining fragments inside the current component: %s",
                ", ".join(sorted(remaining)),
            )
            break

        target_world_pcd = _aggregate_placed_targets(fragments, transformations)
        voxel_sizes = [2.0, 1.0, 0.5]
        best_candidate: Dict[str, Any] | None = None

        for candidate_match in candidate_matches:
            name_a = str(candidate_match["fragment_a"])
            name_b = str(candidate_match["fragment_b"])
            score = float(candidate_match.get("score", 0.0))
            if name_a in placed:
                placed_name, new_name = name_a, name_b
            else:
                placed_name, new_name = name_b, name_a

            initial_transform = _compose_global_transform(
                placed_name,
                new_name,
                candidate_match,
                transformations,
            )
            refined_world = _refine_world_alignment(
                fragments[new_name],
                target_world_pcd,
                initial_transform=initial_transform,
                voxel_sizes=voxel_sizes,
                acceptance_threshold=ASSEMBLY_EXTENSION_THRESHOLD,
            )
            if bool(refined_world["rejected"]):
                continue

            candidate = {
                "new_name": new_name,
                "placed_name": placed_name,
                "score": score,
                "fitness": float(refined_world["fitness"]),
                "transformation": np.asarray(refined_world["transformation"], dtype=float),
                "seed_score": _combined_seed_score(
                    candidate_match,
                    fragment_diagonals,
                    max_diagonal,
                ),
            }
            if best_candidate is None or (
                float(candidate["fitness"]),
                float(candidate["seed_score"]),
                float(candidate["score"]),
            ) > (
                float(best_candidate["fitness"]),
                float(best_candidate["seed_score"]),
                float(best_candidate["score"]),
            ):
                best_candidate = candidate

        if best_candidate is None:
            LOGGER.warning(
                "No remaining candidate inside the component cleared the assembly fitness threshold %.2f",
                ASSEMBLY_EXTENSION_THRESHOLD,
            )
            break

        new_name = str(best_candidate["new_name"])
        transformations[new_name] = np.asarray(best_candidate["transformation"], dtype=float)
        placed.add(new_name)
        remaining.remove(new_name)
        LOGGER.info(
            "Placed %s using %s with score %.3f and refined fitness %.3f",
            new_name,
            str(best_candidate["placed_name"]),
            float(best_candidate["score"]),
            float(best_candidate["fitness"]),
        )

    if remaining and break_surfaces and features_dict:
        LOGGER.info("Second-pass matching for unplaced large fragments in component")
        from src.matching import match_fpfh

        placed_target_pcd = _aggregate_placed_targets(fragments, transformations)
        for fragment_name in sorted(
            [
                name
                for name in remaining
                if break_surface_sizes.get(name, 0) > SECOND_PASS_BREAK_POINT_THRESHOLD
            ],
            key=lambda name: (
                fragment_diagonals.get(name, 0.0),
                break_surface_sizes.get(name, 0),
            ),
            reverse=True,
        ):
            best_candidate = None
            source_break = _get_break_pointcloud(break_surfaces.get(fragment_name))
            source_features = features_dict.get(fragment_name)
            if source_break is None or source_break.is_empty() or source_features is None:
                continue

            for placed_name in sorted(
                placed,
                key=lambda name: fragment_diagonals.get(name, 0.0),
                reverse=True,
            ):
                target_break = _get_break_pointcloud(break_surfaces.get(placed_name))
                target_features = features_dict.get(placed_name)
                if target_break is None or target_break.is_empty() or target_features is None:
                    continue

                try:
                    source_voxel_size = float(source_features.get("fpfh_voxel_size", 0.5))
                    target_voxel_size = float(target_features.get("fpfh_voxel_size", 0.5))
                    reduced_source_voxel = max(source_voxel_size * 0.75, 1e-3)
                    reduced_target_voxel = max(target_voxel_size * 0.75, 1e-3)
                    ransac_result = match_fpfh(
                        source_features["downsampled_pcd"],
                        target_features["downsampled_pcd"],
                        source_features["fpfh"],
                        target_features["fpfh"],
                        voxel_size=max(reduced_source_voxel, reduced_target_voxel),
                        source_voxel_size=reduced_source_voxel,
                        target_voxel_size=reduced_target_voxel,
                    )
                    correspondence_count = int(len(ransac_result.correspondence_set))
                    if correspondence_count < MIN_RANSAC_CORRESPONDENCES:
                        continue

                    initial_transform = transformations[placed_name] @ np.asarray(
                        ransac_result.transformation,
                        dtype=float,
                    )
                    refined_world = _refine_world_alignment(
                        fragments[fragment_name],
                        placed_target_pcd,
                        initial_transform=initial_transform,
                        voxel_sizes=[2.0, 1.0, 0.5],
                        acceptance_threshold=SECOND_PASS_FITNESS_THRESHOLD,
                    )
                    if bool(refined_world["rejected"]):
                        continue

                    candidate = {
                        "placed_name": placed_name,
                        "transformation": np.asarray(refined_world["transformation"], dtype=float),
                        "fitness": float(refined_world["fitness"]),
                        "correspondence_count": correspondence_count,
                    }
                    if best_candidate is None or (
                        float(candidate["fitness"]),
                        int(candidate["correspondence_count"]),
                    ) > (
                        float(best_candidate["fitness"]),
                        int(best_candidate["correspondence_count"]),
                    ):
                        best_candidate = candidate
                except Exception:
                    continue

            if best_candidate is None:
                continue

            transformations[fragment_name] = np.asarray(best_candidate["transformation"], dtype=float)
            placed.add(fragment_name)
            remaining.remove(fragment_name)
            placed_target_pcd = _aggregate_placed_targets(fragments, transformations)
            LOGGER.info(
                "Second-pass placed %s using %s with refined fitness %.3f",
                fragment_name,
                str(best_candidate["placed_name"]),
                float(best_candidate["fitness"]),
            )

    return {
        "transformations": transformations,
        "placed": sorted(placed),
        "unplaced": sorted(remaining),
    }


def _attempt_merge_assemblies(
    fragments: Dict[str, GeometryType],
    primary_assembly: Dict[str, Any],
    secondary_assembly: Dict[str, Any],
    break_surfaces: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ],
    features_dict: Dict[str, Dict[str, object]],
) -> Dict[str, Any] | None:
    """Try to merge two disconnected assemblies with a cross-assembly fragment pair."""
    from src.matching import match_fpfh

    primary_transforms = {
        name: np.asarray(transform, dtype=float)
        for name, transform in primary_assembly["transformations"].items()
    }
    secondary_transforms = {
        name: np.asarray(transform, dtype=float)
        for name, transform in secondary_assembly["transformations"].items()
    }
    primary_names = sorted(primary_transforms)
    secondary_names = sorted(secondary_transforms)
    if not primary_names or not secondary_names:
        return None

    primary_pcd = _assembly_pointcloud(fragments, primary_transforms, primary_names)
    secondary_pcd = _assembly_pointcloud(fragments, secondary_transforms, secondary_names)
    if primary_pcd.is_empty() or secondary_pcd.is_empty():
        return None

    best_merge: Dict[str, Any] | None = None
    for primary_name in primary_names:
        target_break = _get_break_pointcloud(break_surfaces.get(primary_name))
        target_features = features_dict.get(primary_name)
        if target_break is None or target_break.is_empty() or target_features is None:
            continue

        for secondary_name in secondary_names:
            source_break = _get_break_pointcloud(break_surfaces.get(secondary_name))
            source_features = features_dict.get(secondary_name)
            if source_break is None or source_break.is_empty() or source_features is None:
                continue

            try:
                source_voxel_size = float(source_features.get("fpfh_voxel_size", 0.5))
                target_voxel_size = float(target_features.get("fpfh_voxel_size", 0.5))
                ransac_result = match_fpfh(
                    source_features["downsampled_pcd"],
                    target_features["downsampled_pcd"],
                    source_features["fpfh"],
                    target_features["fpfh"],
                    voxel_size=max(source_voxel_size, target_voxel_size),
                    source_voxel_size=source_voxel_size,
                    target_voxel_size=target_voxel_size,
                )
                correspondence_count = int(len(ransac_result.correspondence_set))
                if correspondence_count < MIN_RANSAC_CORRESPONDENCES:
                    continue

                assembly_offset = (
                    primary_transforms[primary_name]
                    @ np.asarray(ransac_result.transformation, dtype=float)
                    @ np.linalg.inv(secondary_transforms[secondary_name])
                )
                refined_merge = _refine_world_alignment(
                    secondary_pcd,
                    primary_pcd,
                    initial_transform=assembly_offset,
                    voxel_sizes=[2.0, 1.0, 0.5],
                    acceptance_threshold=MERGE_FITNESS_THRESHOLD,
                )
                if bool(refined_merge["rejected"]):
                    continue

                candidate = {
                    "transformation": np.asarray(refined_merge["transformation"], dtype=float),
                    "fitness": float(refined_merge["fitness"]),
                    "correspondence_count": correspondence_count,
                    "primary_name": primary_name,
                    "secondary_name": secondary_name,
                }
                if best_merge is None or (
                    float(candidate["fitness"]),
                    int(candidate["correspondence_count"]),
                ) > (
                    float(best_merge["fitness"]),
                    int(best_merge["correspondence_count"]),
                ):
                    best_merge = candidate
            except Exception:
                continue

    if best_merge is None:
        return None

    merged_transforms = dict(primary_transforms)
    merge_transform = np.asarray(best_merge["transformation"], dtype=float)
    for name, transform in secondary_transforms.items():
        merged_transforms[name] = merge_transform @ np.asarray(transform, dtype=float)

    merged_group = sorted(primary_names + secondary_names)
    return {
        "transformations": merged_transforms,
        "placed": merged_group,
        "unplaced": [],
        "fitness": float(best_merge["fitness"]),
        "primary_name": str(best_merge["primary_name"]),
        "secondary_name": str(best_merge["secondary_name"]),
    }


def global_registration(
    fragments: Dict[str, GeometryType],
    pairwise_matches: List[Dict[str, Any]],
    break_surface_sizes: Dict[str, int] | None = None,
    break_surfaces: Dict[
        str, o3d.geometry.PointCloud | Tuple[o3d.geometry.PointCloud, np.ndarray]
    ]
    | None = None,
    features_dict: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, Any]:
    """Assemble fragments component-by-component, then try to merge assemblies."""
    if not fragments:
        return {
            "transformations": {},
            "assembly_groups": [],
            "merged_assemblies": True,
        }

    break_surface_sizes = break_surface_sizes or {
        name: len(_pointcloud_from_geometry(geometry, num_points=12000).points)
        for name, geometry in fragments.items()
    }
    fragment_diagonals = {
        name: _geometry_diagonal(geometry)
        for name, geometry in fragments.items()
    }
    max_diagonal = max(fragment_diagonals.values(), default=1.0)
    accepted_matches = [
        match
        for match in pairwise_matches
        if not bool(match.get("rejected", False))
        and float(match.get("refined_fitness", 0.0)) >= PAIR_REFINED_FITNESS_THRESHOLD
    ]

    if not accepted_matches:
        first_name = max(
            fragments,
            key=lambda name: fragment_diagonals.get(name, 0.0),
        )
        LOGGER.warning(
            "No refined pairwise matches survived filtering; anchoring only %s at identity",
            first_name,
        )
        return {
            "transformations": {first_name: np.eye(4)},
            "assembly_groups": [[first_name]],
            "merged_assemblies": True,
        }

    assemblies: List[Dict[str, Any]] = []
    remaining_fragments = set(fragments)
    while remaining_fragments:
        remaining_matches = [
            match
            for match in accepted_matches
            if str(match["fragment_a"]) in remaining_fragments
            and str(match["fragment_b"]) in remaining_fragments
        ]
        if not remaining_matches:
            break

        component_records: List[Dict[str, Any]] = []
        for component_names in _build_connected_components(
            sorted(remaining_fragments),
            remaining_matches,
        ):
            component_match_list = [
                match
                for match in remaining_matches
                if str(match["fragment_a"]) in component_names
                and str(match["fragment_b"]) in component_names
            ]
            if not component_match_list:
                continue

            component_seed_matches = [
                match
                for match in component_match_list
                if float(match.get("refined_fitness", 0.0)) >= SEED_FITNESS_THRESHOLD
            ]
            seed_pool = component_seed_matches if component_seed_matches else component_match_list
            best_seed_score = max(
                (
                    _combined_seed_score(match, fragment_diagonals, max_diagonal)
                    for match in seed_pool
                ),
                default=0.0,
            )
            total_size = float(
                sum(fragment_diagonals.get(name, 0.0) for name in component_names)
            )
            component_records.append(
                {
                    "names": component_names,
                    "matches": component_match_list,
                    "best_seed_score": best_seed_score,
                    "total_size": total_size,
                }
            )

        if not component_records:
            break

        component_records.sort(
            key=lambda item: (
                float(item["best_seed_score"]),
                float(item["total_size"]),
                len(item["names"]),
            ),
            reverse=True,
        )
        component_record = component_records[0]
        assembly = _build_component_assembly(
            fragments,
            component_record["names"],
            component_record["matches"],
            fragment_diagonals=fragment_diagonals,
            max_diagonal=max_diagonal,
            break_surface_sizes=break_surface_sizes,
            break_surfaces=break_surfaces,
            features_dict=features_dict,
        )
        if assembly["placed"]:
            assemblies.append(assembly)
            remaining_fragments -= set(assembly["placed"])
        else:
            break

    if not assemblies:
        first_name = max(fragments, key=lambda name: fragment_diagonals.get(name, 0.0))
        return {
            "transformations": {first_name: np.eye(4)},
            "assembly_groups": [[first_name]],
            "merged_assemblies": True,
        }

    final_assemblies: List[Dict[str, Any]] = [assemblies[0]]
    for assembly in assemblies[1:]:
        merged = None
        if break_surfaces and features_dict:
            merged = _attempt_merge_assemblies(
                fragments,
                final_assemblies[0],
                assembly,
                break_surfaces=break_surfaces,
                features_dict=features_dict,
            )

        if merged is None:
            final_assemblies.append(assembly)
            continue

        final_assemblies[0] = merged
        LOGGER.info(
            "Merged disconnected assembly via %s vs %s with refined fitness %.3f",
            str(merged["secondary_name"]),
            str(merged["primary_name"]),
            float(merged["fitness"]),
        )

    transformations: Dict[str, np.ndarray] = {}
    assembly_groups: List[List[str]] = []
    for assembly in final_assemblies:
        group_names = sorted(
            name
            for name in assembly["placed"]
            if name in assembly["transformations"]
        )
        if not group_names:
            continue
        assembly_groups.append(group_names)
        for name in group_names:
            transformations[name] = np.asarray(assembly["transformations"][name], dtype=float)

    if len(assembly_groups) > 1:
        LOGGER.warning(
            "Built %d disconnected assemblies that could not be merged cleanly",
            len(assembly_groups),
        )

    remaining = set(fragments) - set(transformations)
    for leftover in sorted(remaining):
        LOGGER.warning(
            "Fragment %s could not be connected to the current assembly and will be left out",
            leftover,
        )

    return {
        "transformations": transformations,
        "assembly_groups": assembly_groups,
        "merged_assemblies": len(assembly_groups) <= 1,
    }


def assemble_reconstruction(
    fragments: Dict[str, o3d.geometry.TriangleMesh],
    transformations: Dict[str, np.ndarray],
    output_dir: Path = OUTPUT_DIR,
    output_name: str = "assembled_reconstruction.ply",
    compute_metrics: bool = True,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, float | None]]:
    """Apply final poses, save the assembled result, and compute summary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    assembled_mesh = o3d.geometry.TriangleMesh()
    transformed_meshes: Dict[str, o3d.geometry.TriangleMesh] = {}

    for name, mesh in fragments.items():
        if name not in transformations:
            continue

        mesh_copy = _copy_mesh(mesh)
        mesh_copy.transform(transformations[name])
        mesh_copy.compute_vertex_normals()
        transformed_meshes[name] = mesh_copy
        assembled_mesh += mesh_copy

    assembled_mesh.compute_vertex_normals()

    if compute_metrics:
        gap_volume, overall_rmse = _estimate_gap_volume_and_rmse(transformed_meshes)
    else:
        gap_volume, overall_rmse = None, None
    # This is a fragment-level proxy for how much of the break-surface evidence
    # made it into the final assembly. We can replace it with a true surface-level
    # metric once break-surface correspondences are tracked explicitly.
    matched_break_surface_percentage = (
        100.0 * len(transformations) / max(len(fragments), 1)
    )
    # TODO: Replace this fragment-level proxy with a true surface-coverage metric.

    metrics: Dict[str, float | None] = {
        "total_gap_volume_estimate": None if gap_volume is None else float(gap_volume),
        "matched_break_surface_percentage": float(matched_break_surface_percentage),
        "overall_alignment_rmse": None if overall_rmse is None else float(overall_rmse),
    }

    output_path = Path(output_dir) / output_name
    o3d.io.write_triangle_mesh(str(output_path), assembled_mesh)
    LOGGER.info("Saved assembled reconstruction to %s", output_path)

    return assembled_mesh, metrics
