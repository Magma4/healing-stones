"""Geometric feature extraction utilities.

FPFH descriptors are the main workhorses for matching break surfaces. The
curvature histogram, surface profile, and shape-context features are meant to
supplement FPFH when fragments have gaps, size differences, or weaker local
correspondence.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from src.config import (
    CURVATURE_NEIGHBORS,
    FEATURE_MAX_POINTS,
    FPFH_RADIUS,
    LOGGER,
)
from src.data_loader import mesh_to_pointcloud


EPSILON = 1e-8


def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh, num_points: int | None = None
) -> o3d.geometry.PointCloud:
    """Compatibility wrapper that samples a point cloud from a mesh."""
    point_count = num_points if num_points is not None else max(1000, len(mesh.vertices))
    return mesh_to_pointcloud(mesh, num_points=point_count)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors row-wise while avoiding division by zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return vectors / norms


def _make_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis spanning the tangent plane of a normal."""
    norm = np.linalg.norm(normal)
    if norm < EPSILON:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / norm

    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    tangent_a = np.cross(normal, reference)
    tangent_a = tangent_a / max(np.linalg.norm(tangent_a), EPSILON)
    tangent_b = np.cross(normal, tangent_a)
    tangent_b = tangent_b / max(np.linalg.norm(tangent_b), EPSILON)
    return tangent_a, tangent_b


def _cap_pointcloud(
    pcd: o3d.geometry.PointCloud,
    max_points: int = FEATURE_MAX_POINTS,
) -> o3d.geometry.PointCloud:
    """Keep very large point clouds at a manageable size for descriptor extraction."""
    if len(pcd.points) <= max_points:
        return o3d.geometry.PointCloud(pcd)

    stride = max(int(np.ceil(len(pcd.points) / float(max_points))), 1)
    return pcd.uniform_down_sample(every_k_points=stride)


def estimate_feature_voxel_size(
    pcd: o3d.geometry.PointCloud, base_voxel_size: float = 0.5
) -> float:
    """Choose a fragment-relative voxel size from the break-surface diagonal."""
    if pcd.is_empty():
        raise ValueError("Cannot estimate voxel size for an empty point cloud")

    bbox_diagonal = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    # `base_voxel_size` is kept for compatibility, but fragment scale drives the descriptor radius now.
    adaptive_voxel_size = max(bbox_diagonal / 50.0, 1e-3)
    return float(adaptive_voxel_size)


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.5,
    fragment_name: str | None = None,
) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample a point cloud and compute FPFH descriptors."""
    if pcd.is_empty():
        raise ValueError("Cannot compute FPFH features for an empty point cloud")

    effective_voxel_size = estimate_feature_voxel_size(pcd, base_voxel_size=voxel_size)
    bbox_diagonal = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    normal_radius = effective_voxel_size * 2.0
    fpfh_radius = effective_voxel_size * 5.0
    if fragment_name is not None:
        LOGGER.info(
            "%s adaptive FPFH scale: diagonal=%.2f, voxel_size=%.2f",
            fragment_name,
            bbox_diagonal,
            effective_voxel_size,
        )
    else:
        LOGGER.info(
            "Adaptive FPFH scale: diagonal=%.2f, voxel_size=%.2f",
            bbox_diagonal,
            effective_voxel_size,
        )
    working_pcd = _cap_pointcloud(pcd, max_points=FEATURE_MAX_POINTS)

    downsampled_pcd = working_pcd.voxel_down_sample(effective_voxel_size)
    if downsampled_pcd.is_empty():
        LOGGER.warning("Voxel downsampling removed all points; falling back to original cloud")
        downsampled_pcd = working_pcd

    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius,
        max_nn=30,
    )
    downsampled_pcd.estimate_normals(normal_search)
    downsampled_pcd.normalize_normals()

    feature_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=fpfh_radius,
        max_nn=100,
    )
    fpfh_features = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled_pcd,
        feature_search,
    )
    return downsampled_pcd, fpfh_features


def compute_curvature_histogram(
    pcd: o3d.geometry.PointCloud, k: int = 30, num_bins: int = 20
) -> np.ndarray:
    """Compute a normalized histogram of local point-cloud curvature values."""
    if pcd.is_empty():
        raise ValueError("Cannot compute curvature histogram for an empty point cloud")

    points = np.asarray(pcd.points)
    if len(points) < 3:
        raise ValueError("Need at least three points to estimate curvature")

    neighbor_count = min(max(k, 3), len(points))
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(points), dtype=float)

    for point_index, point in enumerate(points):
        _, neighbor_indices, _ = tree.search_knn_vector_3d(point, neighbor_count)
        if len(neighbor_indices) < 3:
            continue

        local_points = points[neighbor_indices]
        centered = local_points - local_points.mean(axis=0)
        covariance = np.cov(centered.T, bias=True)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
        curvatures[point_index] = float(eigenvalues[0] / (eigenvalues.sum() + EPSILON))

    histogram, _ = np.histogram(curvatures, bins=num_bins, range=(0.0, 1.0))
    histogram = histogram.astype(float)
    histogram /= histogram.sum() + EPSILON
    return histogram


def compute_surface_profile(
    pcd: o3d.geometry.PointCloud, direction: str = "principal", num_samples: int = 100
) -> np.ndarray:
    """Project the cloud into its principal plane and sample a 2D profile curve."""
    if pcd.is_empty():
        raise ValueError("Cannot compute a surface profile for an empty point cloud")

    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1")

    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    principal_axis = eigenvectors[:, order[0]]
    secondary_axis = eigenvectors[:, order[1]]

    if direction == "principal":
        distances = centered @ principal_axis
        heights = centered @ secondary_axis
    elif direction == "secondary":
        distances = centered @ secondary_axis
        heights = centered @ principal_axis
    else:
        raise ValueError("direction must be 'principal' or 'secondary'")

    distance_min = float(distances.min())
    distance_max = float(distances.max())
    if abs(distance_max - distance_min) < EPSILON:
        return np.column_stack(
            [np.zeros(num_samples, dtype=float), np.zeros(num_samples, dtype=float)]
        )

    bin_edges = np.linspace(distance_min, distance_max, num_samples + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sampled_heights = np.full(num_samples, np.nan, dtype=float)

    for bin_index in range(num_samples):
        start = bin_edges[bin_index]
        end = bin_edges[bin_index + 1]
        if bin_index == num_samples - 1:
            mask = (distances >= start) & (distances <= end)
        else:
            mask = (distances >= start) & (distances < end)

        if np.any(mask):
            sampled_heights[bin_index] = float(np.mean(heights[mask]))

    valid_indices = np.flatnonzero(~np.isnan(sampled_heights))
    if len(valid_indices) == 0:
        sampled_heights[:] = 0.0
    elif len(valid_indices) == 1:
        sampled_heights[:] = sampled_heights[valid_indices[0]]
    else:
        missing_indices = np.flatnonzero(np.isnan(sampled_heights))
        sampled_heights[missing_indices] = np.interp(
            bin_centers[missing_indices],
            bin_centers[valid_indices],
            sampled_heights[valid_indices],
        )

    return np.column_stack([bin_centers, sampled_heights])


def compute_shape_context(
    pcd: o3d.geometry.PointCloud, num_bins_r: int = 5, num_bins_theta: int = 12
) -> np.ndarray:
    """Compute a simplified tangent-plane shape-context descriptor per point."""
    if pcd.is_empty():
        raise ValueError("Cannot compute shape context for an empty point cloud")

    if num_bins_r <= 0 or num_bins_theta <= 0:
        raise ValueError("num_bins_r and num_bins_theta must be positive")

    working_pcd = o3d.geometry.PointCloud(pcd)
    if not working_pcd.has_normals() or len(working_pcd.normals) != len(working_pcd.points):
        bbox_extent = np.linalg.norm(working_pcd.get_axis_aligned_bounding_box().get_extent())
        normal_radius = max(bbox_extent * 0.05, 1e-3)
        working_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
        working_pcd.normalize_normals()

    points = np.asarray(working_pcd.points)
    normals = _normalize_vectors(np.asarray(working_pcd.normals))
    point_count = len(points)
    neighbor_count = min(64, point_count)
    tree = o3d.geometry.KDTreeFlann(working_pcd)

    bbox_extent = np.linalg.norm(working_pcd.get_axis_aligned_bounding_box().get_extent())
    max_radius = max(bbox_extent, 1e-3)
    min_radius = max(max_radius * 1e-3, 1e-5)
    radial_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), num_bins_r + 1)
    theta_edges = np.linspace(0.0, 2.0 * np.pi, num_bins_theta + 1)

    descriptors = np.zeros((point_count, num_bins_r * num_bins_theta), dtype=float)

    for point_index, point in enumerate(points):
        _, neighbor_indices, _ = tree.search_knn_vector_3d(point, neighbor_count)
        neighbor_indices = [idx for idx in neighbor_indices if idx != point_index]
        if not neighbor_indices:
            continue

        local_vectors = points[neighbor_indices] - point
        tangent_a, tangent_b = _make_tangent_basis(normals[point_index])
        projected_x = local_vectors @ tangent_a
        projected_y = local_vectors @ tangent_b
        radii = np.sqrt(projected_x**2 + projected_y**2)
        theta = (np.arctan2(projected_y, projected_x) + 2.0 * np.pi) % (2.0 * np.pi)

        histogram, _, _ = np.histogram2d(
            radii,
            theta,
            bins=[radial_edges, theta_edges],
        )
        descriptor = histogram.ravel().astype(float)
        descriptor /= descriptor.sum() + EPSILON
        descriptors[point_index] = descriptor

    return descriptors


def extract_all_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.5,
    fragment_name: str | None = None,
) -> Dict[str, object]:
    """Compute the main feature bundle used by fragment matching."""
    effective_voxel_size = estimate_feature_voxel_size(pcd, base_voxel_size=voxel_size)
    downsampled_pcd, fpfh_features = compute_fpfh_features(
        pcd,
        voxel_size=voxel_size,
        fragment_name=fragment_name,
    )

    return {
        "downsampled_pcd": downsampled_pcd,
        "fpfh": fpfh_features,
        "fpfh_matrix": np.asarray(fpfh_features.data).T,
        "fpfh_voxel_size": effective_voxel_size,
        "break_point_count": int(len(pcd.points)),
        "break_bbox_diagonal": float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())),
        "curvature_histogram": compute_curvature_histogram(
            downsampled_pcd, k=CURVATURE_NEIGHBORS
        ),
        "surface_profile": compute_surface_profile(downsampled_pcd),
        "shape_context": compute_shape_context(downsampled_pcd),
    }


def compute_fpfh(
    pcd: o3d.geometry.PointCloud, radius: float = FPFH_RADIUS
) -> o3d.pipelines.registration.Feature:
    """Compatibility helper that computes FPFH on the given point cloud directly."""
    if pcd.is_empty():
        raise ValueError("Cannot compute FPFH features for an empty point cloud")

    working_pcd = o3d.geometry.PointCloud(pcd)
    if not working_pcd.has_normals() or len(working_pcd.normals) != len(working_pcd.points):
        working_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 0.4, max_nn=30)
        )
        working_pcd.normalize_normals()

    search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    return o3d.pipelines.registration.compute_fpfh_feature(working_pcd, search)


def compute_basic_descriptors(mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
    """Return simple global descriptors as a starting point."""
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0) if len(vertices) else np.zeros(3)
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    return {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "curvature_neighbors": float(CURVATURE_NEIGHBORS),
    }
