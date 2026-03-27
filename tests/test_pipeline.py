"""Synthetic regression tests for the cleanup-safe pipeline surface."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_fragment, preprocess_mesh
from src.matching import compute_match_score
from src.surface_classifier import compute_surface_features


def _synthetic_mesh() -> o3d.geometry.TriangleMesh:
    """Build a small translated mesh so preprocessing has visible work to do."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    mesh.compute_vertex_normals()
    mesh.translate((3.0, -2.0, 5.0))
    return mesh


def test_load_fragment_raises_for_missing_file(tmp_path) -> None:
    """Missing fragment paths should raise a clear file error."""
    missing_path = tmp_path / "missing_fragment.ply"
    try:
        load_fragment(missing_path)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("load_fragment should raise FileNotFoundError for a missing mesh")


def test_preprocess_centers_mesh_near_origin() -> None:
    """Preprocessing should recentre the mesh without requiring real artifact data."""
    processed = preprocess_mesh(_synthetic_mesh())
    centroid = np.asarray(processed.vertices).mean(axis=0)
    assert np.allclose(centroid, np.zeros(3), atol=1e-6)


def test_surface_features_have_expected_shape() -> None:
    """Surface feature extraction should return one 5D feature vector per point."""
    mesh = _synthetic_mesh()
    point_cloud = mesh.sample_points_uniformly(number_of_points=256)
    features = compute_surface_features(mesh=None, pcd=point_cloud)
    assert features.shape == (len(point_cloud.points), 5)


def test_match_score_stays_between_zero_and_one() -> None:
    """Pairwise match scoring should remain normalized."""
    mesh = _synthetic_mesh()
    source_pcd = mesh.sample_points_uniformly(number_of_points=256)
    target_pcd = o3d.geometry.PointCloud(source_pcd)

    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        0.25,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    score = compute_match_score(reg_result, source_pcd, target_pcd)
    assert 0.0 <= score <= 1.0


def test_match_score_is_higher_for_identity_than_bad_translation() -> None:
    """A good alignment should not score worse than an obviously bad one."""
    mesh = _synthetic_mesh()
    source_pcd = mesh.sample_points_uniformly(number_of_points=256)
    target_pcd = o3d.geometry.PointCloud(source_pcd)

    good_result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        0.25,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    bad_result = SimpleNamespace(
        transformation=np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        fitness=0.0,
        inlier_rmse=float("inf"),
    )

    good_score = compute_match_score(good_result, source_pcd, target_pcd)
    bad_score = compute_match_score(bad_result, source_pcd, target_pcd)
    assert good_score >= bad_score
