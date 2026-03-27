"""Mesh loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d

from src.config import DATA_DIR, LOGGER


SUPPORTED_EXTENSIONS = {".ply", ".obj"}


def list_fragment_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return supported fragment files found in the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        LOGGER.warning("Data directory does not exist: %s", data_dir)
        return []

    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    LOGGER.info("Found %d fragment file(s)", len(files))
    return files


def load_fragment(filepath: str | Path) -> o3d.geometry.TriangleMesh:
    """Load a single `.ply` or `.obj` fragment as an Open3D triangle mesh."""
    path = Path(filepath)

    if not path.exists():
        LOGGER.error("Fragment file not found: %s", path)
        raise FileNotFoundError(f"Fragment file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        LOGGER.error("Unsupported fragment format: %s", path.suffix)
        raise ValueError(f"Unsupported fragment format: {path.suffix}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        LOGGER.error("Loaded empty mesh from %s", path)
        raise ValueError(f"Loaded empty mesh from {path}")

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        LOGGER.error("Mesh has no usable geometry: %s", path)
        raise ValueError(f"Mesh has no usable geometry: {path}")

    if not mesh.has_vertex_normals() or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.compute_vertex_normals()

    return mesh


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove common mesh issues before downstream processing."""
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def preprocess_mesh(
    mesh: o3d.geometry.TriangleMesh, voxel_size: float | None = None
) -> o3d.geometry.TriangleMesh:
    """Clean, center, and optionally simplify a mesh fragment."""
    if mesh.is_empty():
        raise ValueError("Cannot preprocess an empty mesh")

    processed = o3d.geometry.TriangleMesh(mesh)
    processed = _cleanup_mesh(processed)

    if len(processed.vertices) == 0 or len(processed.triangles) == 0:
        raise ValueError("Mesh became empty after cleanup")

    centroid = np.asarray(processed.vertices).mean(axis=0)
    processed.translate(-centroid)

    if voxel_size is not None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive when provided")
        processed = processed.simplify_vertex_clustering(voxel_size=float(voxel_size))
        processed = _cleanup_mesh(processed)

    if processed.is_empty() or len(processed.vertices) == 0 or len(processed.triangles) == 0:
        raise ValueError("Processed mesh has no usable geometry")

    processed.compute_vertex_normals()
    return processed


def mesh_to_pointcloud(
    mesh: o3d.geometry.TriangleMesh, num_points: int = 50000
) -> o3d.geometry.PointCloud:
    """Sample a point cloud uniformly from the mesh surface."""
    if num_points <= 0:
        raise ValueError("num_points must be a positive integer")

    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError("Cannot sample points from an empty mesh")

    pointcloud = mesh.sample_points_uniformly(number_of_points=int(num_points))
    if pointcloud.is_empty():
        raise ValueError("Point cloud sampling produced no points")
    return pointcloud


def compute_mesh_stats(mesh: o3d.geometry.TriangleMesh) -> Dict[str, object]:
    """Return basic geometry statistics for a mesh."""
    if mesh.is_empty():
        return {
            "num_vertices": 0,
            "num_faces": 0,
            "bounding_box": {"x": 0.0, "y": 0.0, "z": 0.0},
            "surface_area": 0.0,
            "is_watertight": False,
        }

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    surface_area = float(mesh.get_surface_area()) if len(mesh.triangles) > 0 else 0.0

    try:
        is_watertight = bool(mesh.is_watertight()) if len(mesh.triangles) > 0 else False
    except RuntimeError:
        is_watertight = False

    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.triangles)),
        "bounding_box": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2]),
        },
        "surface_area": surface_area,
        "is_watertight": is_watertight,
    }


def load_all_fragments(
    data_dir: str | Path = DATA_DIR, voxel_size: float | None = None
) -> Dict[str, o3d.geometry.TriangleMesh]:
    """Load and preprocess all supported fragment meshes in a directory."""
    fragment_dir = Path(data_dir)
    fragment_paths = list_fragment_files(fragment_dir)
    fragments: Dict[str, o3d.geometry.TriangleMesh] = {}
    summaries = []

    for path in fragment_paths:
        try:
            LOGGER.info("Loading fragment: %s", path.name)
            mesh = load_fragment(path)
            processed_mesh = preprocess_mesh(mesh, voxel_size=voxel_size)
            fragments[path.name] = processed_mesh
            summaries.append((path.name, compute_mesh_stats(processed_mesh)))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping fragment %s: %s", path.name, exc)

    print(f"Loaded {len(fragments)} fragment(s) from {fragment_dir}")
    for name, stats in summaries:
        dims = stats["bounding_box"]
        print(
            f"- {name}: vertices={stats['num_vertices']}, "
            f"faces={stats['num_faces']}, "
            f"bbox=({dims['x']:.2f}, {dims['y']:.2f}, {dims['z']:.2f})"
        )

    return fragments


# Backward-compatible alias for older code that still imports `load_mesh`.
load_mesh = load_fragment
