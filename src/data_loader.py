"""Mesh loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from src.config import DATA_DIR, LOGGER, WORKING_MAX_VERTICES


SUPPORTED_EXTENSIONS = {".ply", ".obj"}
PLY_DTYPE_MAP = {
    "char": np.int8,
    "uchar": np.uint8,
    "int8": np.int8,
    "uint8": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int16": np.int16,
    "uint16": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "int32": np.int32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


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


def _load_fragment_raw(filepath: str | Path) -> o3d.geometry.TriangleMesh:
    """Load a mesh without triggering the expensive normal computation pass."""
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

    return mesh


def _read_ply_header(filepath: str | Path) -> Dict[str, object]:
    """Parse enough of a binary PLY header to read vertex data quickly."""
    path = Path(filepath)
    with path.open("rb") as handle:
        first_line = handle.readline().decode("ascii", errors="ignore").strip()
        if first_line != "ply":
            raise ValueError(f"{path} is not a valid PLY file")

        format_name = None
        vertex_count = None
        face_count = 0
        vertex_properties: List[tuple[str, str]] = []
        current_element = None

        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected end of PLY header in {path}")

            decoded = line.decode("ascii", errors="ignore").strip()
            if decoded == "end_header":
                break
            if not decoded or decoded.startswith("comment"):
                continue

            parts = decoded.split()
            keyword = parts[0]
            if keyword == "format":
                format_name = parts[1]
            elif keyword == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    face_count = int(parts[2])
            elif keyword == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("List properties in vertex elements are not supported")
                vertex_properties.append((parts[2], parts[1]))

        header_end = handle.tell()

    if format_name != "binary_little_endian":
        raise ValueError(f"Only binary little-endian PLY is supported, got {format_name}")
    if vertex_count is None:
        raise ValueError(f"PLY header did not include a vertex count: {path}")
    if not vertex_properties:
        raise ValueError(f"PLY header did not include vertex properties: {path}")

    dtype = np.dtype(
        [(name, PLY_DTYPE_MAP[dtype_name]) for name, dtype_name in vertex_properties]
    )
    return {
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
        "vertex_dtype": dtype,
        "header_end": int(header_end),
    }


def _load_ply_vertices_fast(filepath: str | Path) -> tuple[np.ndarray, int, int]:
    """Load only vertex positions from a binary PLY without reading faces."""
    path = Path(filepath)
    header = _read_ply_header(path)
    dtype = header["vertex_dtype"]
    vertex_count = int(header["vertex_count"])
    face_count = int(header["face_count"])
    header_end = int(header["header_end"])

    with path.open("rb") as handle:
        handle.seek(header_end)
        vertex_data = np.fromfile(handle, dtype=dtype, count=vertex_count)

    if len(vertex_data) != vertex_count:
        raise ValueError(f"Could not read all PLY vertices from {path}")

    required = {"x", "y", "z"}
    if not required.issubset(vertex_data.dtype.names or ()):
        raise ValueError(f"PLY file {path} is missing x/y/z vertex properties")

    points = np.column_stack(
        [
            np.asarray(vertex_data["x"], dtype=np.float32),
            np.asarray(vertex_data["y"], dtype=np.float32),
            np.asarray(vertex_data["z"], dtype=np.float32),
        ]
    )
    return points, vertex_count, face_count


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


def center_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Center a mesh at the origin without doing heavy cleanup on the full mesh."""
    if mesh.is_empty() or len(mesh.vertices) == 0:
        raise ValueError("Cannot center an empty mesh")

    centered = o3d.geometry.TriangleMesh(mesh)
    centroid = np.asarray(centered.vertices).mean(axis=0)
    centered.translate(-centroid)
    return centered


def _pointcloud_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from a numpy array of xyz points."""
    if len(points) == 0:
        raise ValueError("Cannot build a point cloud from zero points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def center_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Center a point cloud at the origin."""
    if pcd.is_empty() or len(pcd.points) == 0:
        raise ValueError("Cannot center an empty point cloud")

    centered = o3d.geometry.PointCloud(pcd)
    centroid = np.asarray(centered.points).mean(axis=0)
    centered.translate(-centroid)
    return centered


def _mesh_vertices_to_pointcloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    """Convert mesh vertices into a point cloud while preserving normals when possible."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    return pcd


def _estimate_working_voxel_size_from_pcd(
    pcd: o3d.geometry.PointCloud,
    target_vertices: int,
) -> float:
    """Estimate the initial voxel size from the fragment's bounding-box diagonal."""
    bbox_diagonal = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    return max(
        bbox_diagonal / (float(target_vertices) ** (1.0 / 3.0)),
        1e-3,
    )


def _downsample_pointcloud_with_details(
    pcd: o3d.geometry.PointCloud,
    target_vertices: int = WORKING_MAX_VERTICES,
) -> tuple[o3d.geometry.PointCloud, float]:
    """Convert a dense point cloud into a smaller working-resolution point cloud.

    This is the main fast-path used by the pipeline. For huge PLY files we can
    skip reading faces entirely and downsample directly from the vertices.
    """
    if pcd.is_empty() or len(pcd.points) == 0:
        raise ValueError("Cannot downsample an empty point cloud")

    if target_vertices <= 0:
        raise ValueError("target_vertices must be positive")

    working_pcd = center_pointcloud(pcd)
    if len(working_pcd.points) <= target_vertices:
        if not working_pcd.has_normals() or len(working_pcd.normals) != len(working_pcd.points):
            bbox_diagonal = float(
                np.linalg.norm(working_pcd.get_axis_aligned_bounding_box().get_extent())
            )
            normal_radius = max(bbox_diagonal * 0.02, 1e-3)
            working_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
            )
            working_pcd.normalize_normals()
        return working_pcd, 0.0

    bbox_diagonal = float(np.linalg.norm(working_pcd.get_axis_aligned_bounding_box().get_extent()))
    voxel_size = _estimate_working_voxel_size_from_pcd(working_pcd, target_vertices)
    min_points = min(target_vertices, max(3000, int(target_vertices * 0.15)))
    max_points = target_vertices
    downsampled_pcd = o3d.geometry.PointCloud()

    for _ in range(6):
        candidate = working_pcd.voxel_down_sample(voxel_size)
        if candidate.is_empty():
            voxel_size *= 0.5
            continue

        point_count = len(candidate.points)
        downsampled_pcd = candidate
        if point_count < min_points:
            voxel_size *= 0.65
            continue
        if point_count > int(max_points * 1.2):
            voxel_size *= 1.2
            continue
        break

    if downsampled_pcd.is_empty():
        LOGGER.warning("Voxel downsampling removed all points; falling back to mesh vertices")
        downsampled_pcd = working_pcd

    if len(downsampled_pcd.points) > max_points:
        # The diagonal-based estimate is only a first guess. A light trim keeps
        # the cloud inside the working budget after the adaptive refinement loop.
        stride = max(int(np.ceil(len(downsampled_pcd.points) / float(max_points))), 1)
        downsampled_pcd = downsampled_pcd.uniform_down_sample(every_k_points=stride)

    normal_radius = max(voxel_size * 2.0, bbox_diagonal * 0.02, 1e-3)
    downsampled_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    downsampled_pcd.normalize_normals()
    return downsampled_pcd, voxel_size


def _downsample_mesh_with_details(
    mesh: o3d.geometry.TriangleMesh,
    target_vertices: int = WORKING_MAX_VERTICES,
) -> tuple[o3d.geometry.PointCloud, float]:
    """Compatibility wrapper that downsamples from a mesh via its vertices."""
    return _downsample_pointcloud_with_details(
        _mesh_vertices_to_pointcloud(mesh),
        target_vertices=target_vertices,
    )


def downsample_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_vertices: int = WORKING_MAX_VERTICES,
) -> o3d.geometry.PointCloud:
    """Public wrapper that returns only the working-resolution point cloud."""
    downsampled_pcd, _ = _downsample_mesh_with_details(
        mesh,
        target_vertices=target_vertices,
    )
    return downsampled_pcd


def load_selected_full_meshes(
    fragment_paths: Dict[str, Path],
    fragment_names: List[str] | Tuple[str, ...] | set[str],
) -> Dict[str, o3d.geometry.TriangleMesh]:
    """Load only the full-resolution meshes needed for final output."""
    meshes: Dict[str, o3d.geometry.TriangleMesh] = {}
    for name in fragment_names:
        if name not in fragment_paths:
            continue
        meshes[name] = center_mesh(_load_fragment_raw(fragment_paths[name]))
    return meshes


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
    data_dir: str | Path = DATA_DIR,
    max_vertices: int = WORKING_MAX_VERTICES,
    load_full_meshes: bool = True,
) -> Tuple[
    Dict[str, o3d.geometry.TriangleMesh],
    Dict[str, o3d.geometry.PointCloud],
    Dict[str, Dict[str, object]],
]:
    """Load fragments and return full meshes, working clouds, and metadata."""
    fragment_dir = Path(data_dir)
    fragment_paths = list_fragment_files(fragment_dir)
    meshes_full: Dict[str, o3d.geometry.TriangleMesh] = {}
    meshes_working: Dict[str, o3d.geometry.PointCloud] = {}
    fragment_metadata: Dict[str, Dict[str, object]] = {}
    summaries = []
    seen_signatures: Dict[tuple[int, int], str] = {}

    for path in fragment_paths:
        try:
            LOGGER.info("Loading fragment: %s", path.name)
            if path.suffix.lower() == ".ply":
                points, vertex_count, face_count = _load_ply_vertices_fast(path)
                original_stats = {
                    "num_vertices": int(vertex_count),
                    "num_faces": int(face_count),
                }
                working_source = _pointcloud_from_points(points)
                working_source = center_pointcloud(working_source)
                signature = (int(vertex_count), int(face_count))
            else:
                original_mesh = center_mesh(_load_fragment_raw(path))
                mesh_stats = compute_mesh_stats(original_mesh)
                original_stats = {
                    "num_vertices": int(mesh_stats["num_vertices"]),
                    "num_faces": int(mesh_stats["num_faces"]),
                }
                working_source = _mesh_vertices_to_pointcloud(original_mesh)
                signature = (
                    int(original_stats["num_vertices"]),
                    int(original_stats["num_faces"]),
                )

            if signature in seen_signatures:
                LOGGER.warning(
                    "Skipping possible duplicate %s because it matches vertex/face counts of %s",
                    path.name,
                    seen_signatures[signature],
                )
                continue

            working_pcd, working_voxel_size = _downsample_pointcloud_with_details(
                working_source,
                target_vertices=max_vertices,
            )
            working_bbox = working_pcd.get_axis_aligned_bounding_box()
            working_extent = working_bbox.get_extent()

            if load_full_meshes:
                if path.suffix.lower() == ".ply":
                    meshes_full[path.name] = center_mesh(_load_fragment_raw(path))
                else:
                    meshes_full[path.name] = original_mesh
            meshes_working[path.name] = working_pcd
            fragment_metadata[path.name] = {
                "path": path,
                "original_vertices": int(original_stats["num_vertices"]),
                "original_faces": int(original_stats["num_faces"]),
                "working_points": int(len(working_pcd.points)),
                "working_voxel_size": float(working_voxel_size),
            }
            seen_signatures[signature] = path.name
            summaries.append(
                (
                    path.name,
                    original_stats,
                    int(len(working_pcd.points)),
                    working_voxel_size,
                    working_extent,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping fragment %s: %s", path.name, exc)

    print(f"Loaded {len(meshes_working)} fragment(s) from {fragment_dir}")
    for name, original_stats, working_points, working_voxel_size, working_extent in summaries:
        print(
            f"- {name}: original_vertices={original_stats['num_vertices']}, "
            f"working_points={working_points}, "
            f"faces={original_stats['num_faces']}, "
            f"working_voxel={working_voxel_size:.2f}, "
            f"bbox=({working_extent[0]:.2f}, {working_extent[1]:.2f}, {working_extent[2]:.2f})"
        )

    return meshes_full, meshes_working, fragment_metadata


# Backward-compatible alias for older code that still imports `load_mesh`.
load_mesh = load_fragment
