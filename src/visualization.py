"""Visualization helpers for debugging, analysis, and final presentation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from src.config import LOGGER, OUTPUT_DIR
from src.data_loader import mesh_to_pointcloud


EPSILON = 1e-8


def _copy_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Return a copy of a mesh so coloring and transforms do not mutate inputs."""
    return o3d.geometry.TriangleMesh(mesh)


def _ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create the output directory when needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _mesh_with_normal_colors(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Color a mesh by vertex normals for quick inspection."""
    colored_mesh = _copy_mesh(mesh)
    if not colored_mesh.has_vertex_normals() or len(colored_mesh.vertex_normals) != len(
        colored_mesh.vertices
    ):
        colored_mesh.compute_vertex_normals()

    normals = np.asarray(colored_mesh.vertex_normals)
    colors = 0.5 * (normals + 1.0)
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return colored_mesh


def _paint_mesh(mesh: o3d.geometry.TriangleMesh, color: Sequence[float]) -> o3d.geometry.TriangleMesh:
    """Paint a mesh with a uniform RGB color."""
    colored_mesh = _copy_mesh(mesh)
    colored_mesh.paint_uniform_color(list(color))
    return colored_mesh


def _show_geometries_non_blocking(
    geometries: Sequence[o3d.geometry.Geometry],
    title: str,
    screenshot_path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    frames: int = 45,
) -> Path | None:
    """Open a lightweight Open3D window briefly so the script does not hang."""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=width, height=height, visible=True)
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.96, 0.96, 0.98], dtype=float)
        render_option.point_size = 3.0
        render_option.mesh_show_back_face = True

        for geometry in geometries:
            vis.add_geometry(geometry)

        for _ in range(frames):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)

        if screenshot_path is not None:
            vis.capture_screen_image(str(screenshot_path), do_render=True)

        vis.destroy_window()
        return screenshot_path
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Open3D visualization failed for '%s': %s", title, exc)
        return None


def _resolve_labels_and_confidence(
    labels: np.ndarray | Tuple[np.ndarray, np.ndarray] | Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Support plain labels, (labels, confidence), or dict-style inputs."""
    if isinstance(labels, tuple) and len(labels) == 2:
        label_array = np.asarray(labels[0], dtype=int)
        confidence = np.asarray(labels[1], dtype=float)
        return label_array, confidence

    if isinstance(labels, dict):
        label_array = np.asarray(labels.get("labels"), dtype=int)
        confidence_array = labels.get("confidence")
        confidence = None if confidence_array is None else np.asarray(confidence_array, dtype=float)
        return label_array, confidence

    return np.asarray(labels, dtype=int), None


def _safe_point_sample(
    mesh: o3d.geometry.TriangleMesh, count: int = 1000
) -> o3d.geometry.PointCloud:
    """Sample a point cloud from a mesh with a conservative default size."""
    return mesh_to_pointcloud(mesh, num_points=max(100, count))


def _stone_palette_from_points(points: np.ndarray) -> list[str]:
    """Generate a discrete multitone limestone palette for uncolored point clouds."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, EPSILON)
    normalized = (points - mins) / spans

    palette = np.array(
        [
            [0.97, 0.95, 0.88],  # chalk
            [0.91, 0.84, 0.66],  # pale sand
            [0.84, 0.70, 0.47],  # warm tan
            [0.78, 0.56, 0.33],  # amber
            [0.67, 0.46, 0.26],  # ochre brown
            [0.52, 0.36, 0.22],  # umber
            [0.66, 0.64, 0.58],  # cool stone gray
        ],
        dtype=float,
    )

    height = normalized[:, 2:3]
    sweep = normalized[:, 1:2]
    ridge = normalized[:, 0:1]

    low_freq = 0.5 + 0.5 * np.sin(
        points[:, 0:1] * 0.025 + points[:, 1:2] * 0.019 + points[:, 2:3] * 0.031
    )
    speckle = 0.5 + 0.5 * np.sin(
        points[:, 0:1] * 0.14 + points[:, 1:2] * 0.11 + points[:, 2:3] * 0.17
    )
    fleck = 0.5 + 0.5 * np.sin(
        points[:, 0:1] * 0.33 + points[:, 1:2] * 0.27 + points[:, 2:3] * 0.21
    )

    weight_chalk = 0.38 + 0.38 * height[:, 0]
    weight_sand = 0.30 + 0.22 * sweep[:, 0]
    weight_tan = 0.18 + 0.15 * low_freq[:, 0]
    weight_amber = 0.10 + 0.18 * speckle[:, 0]
    weight_ochre = 0.08 + 0.12 * (1.0 - ridge[:, 0])
    weight_umber = 0.05 + 0.15 * ((fleck[:, 0] > 0.78).astype(float))
    weight_gray = 0.04 + 0.10 * ((low_freq[:, 0] < 0.32).astype(float))

    weights = np.column_stack(
        [
            weight_chalk,
            weight_sand,
            weight_tan,
            weight_amber,
            weight_ochre,
            weight_umber,
            weight_gray,
        ]
    )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPSILON)
    cumulative = np.cumsum(weights, axis=1)

    selector = np.mod(
        0.37 * low_freq[:, 0] + 0.41 * speckle[:, 0] + 0.22 * fleck[:, 0],
        1.0,
    )
    indices = (selector[:, None] > cumulative).sum(axis=1)
    colors = palette[np.clip(indices, 0, len(palette) - 1)]
    return [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in colors]


def _build_correspondence_lines(
    source_mesh: o3d.geometry.TriangleMesh,
    target_mesh: o3d.geometry.TriangleMesh,
    max_pairs: int = 100,
) -> o3d.geometry.LineSet:
    """Create a simple nearest-neighbor correspondence line set."""
    source_pcd = _safe_point_sample(source_mesh, count=1200)
    target_pcd = _safe_point_sample(target_mesh, count=1200)
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    line_set = o3d.geometry.LineSet()
    if len(source_points) == 0 or len(target_points) == 0:
        return line_set

    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    points: List[np.ndarray] = []
    lines: List[List[int]] = []
    colors: List[List[float]] = []

    max_pairs = min(max_pairs, len(source_points))
    sample_indices = np.linspace(0, len(source_points) - 1, max_pairs, dtype=int)

    for source_index in sample_indices:
        source_point = source_points[source_index]
        _, neighbor_indices, _ = target_tree.search_knn_vector_3d(source_point, 1)
        if not neighbor_indices:
            continue

        target_point = target_points[neighbor_indices[0]]
        start_idx = len(points)
        points.extend([source_point, target_point])
        lines.append([start_idx, start_idx + 1])
        colors.append([0.3, 0.3, 0.3])

    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return line_set


def visualize_fragment(
    mesh: o3d.geometry.TriangleMesh,
    title: str = "Fragment",
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Display a single fragment colored by vertex normals."""
    preview_mesh = _mesh_with_normal_colors(mesh)
    screenshot_path = _ensure_output_dir(output_dir) / f"{title.lower().replace(' ', '_')}.png"
    return _show_geometries_non_blocking([preview_mesh], title=title, screenshot_path=screenshot_path)


def visualize_surface_classification(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray | Tuple[np.ndarray, np.ndarray] | Dict[str, np.ndarray],
    title: str = "Surface Classification",
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Color a mesh by surface labels for debugging break/original regions."""
    label_array, confidence = _resolve_labels_and_confidence(labels)
    if len(label_array) != len(mesh.vertices):
        raise ValueError("labels must match the number of mesh vertices")

    classification_mesh = _copy_mesh(mesh)
    colors = np.zeros((len(label_array), 3), dtype=float)
    colors[label_array == 0] = np.array([0.15, 0.35, 0.9])   # original surface
    colors[label_array == 1] = np.array([0.9, 0.2, 0.2])     # break surface

    if confidence is not None:
        if len(confidence) != len(label_array):
            raise ValueError("confidence must match the number of labels")
        colors[confidence < 0.6] = np.array([0.95, 0.8, 0.1])  # uncertain

    classification_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    screenshot_path = _ensure_output_dir(output_dir) / "surface_classification.png"
    return _show_geometries_non_blocking(
        [classification_mesh],
        title=title,
        screenshot_path=screenshot_path,
    )


def visualize_matches(
    fragment_i: o3d.geometry.TriangleMesh,
    fragment_j: o3d.geometry.TriangleMesh,
    transformation: np.ndarray,
    title: str = "Match",
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Show two fragments before and after alignment with simple correspondences."""
    blue = np.array([0.2, 0.45, 0.9], dtype=float)
    orange = np.array([0.95, 0.55, 0.2], dtype=float)

    source_original = _paint_mesh(fragment_i, blue)
    target_original = _paint_mesh(fragment_j, orange)
    source_aligned = _paint_mesh(fragment_i, blue)
    target_aligned = _paint_mesh(fragment_j, orange)
    source_aligned.transform(transformation)

    extent_i = fragment_i.get_axis_aligned_bounding_box().get_extent()
    extent_j = fragment_j.get_axis_aligned_bounding_box().get_extent()
    spacing = max(float(np.linalg.norm(np.maximum(extent_i, extent_j))), 1.0) * 1.5

    source_original.translate(np.array([-spacing, 0.0, 0.0]))
    target_original.translate(np.array([spacing, 0.0, 0.0]))

    aligned_offset = np.array([3.0 * spacing, 0.0, 0.0], dtype=float)
    source_aligned.translate(aligned_offset)
    target_aligned.translate(aligned_offset)

    correspondences = _build_correspondence_lines(source_aligned, target_aligned, max_pairs=100)
    screenshot_path = _ensure_output_dir(output_dir) / "match_visualization.png"

    return _show_geometries_non_blocking(
        [source_original, target_original, source_aligned, target_aligned, correspondences],
        title=title,
        screenshot_path=screenshot_path,
    )


def visualize_reconstruction(
    assembled_mesh: o3d.geometry.TriangleMesh,
    individual_fragments: Dict[str, o3d.geometry.TriangleMesh] | Sequence[o3d.geometry.TriangleMesh] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Display the assembled reconstruction and save a screenshot."""
    geometries: List[o3d.geometry.Geometry] = []

    if individual_fragments:
        if isinstance(individual_fragments, dict):
            fragment_list = list(individual_fragments.values())
        else:
            fragment_list = list(individual_fragments)

        cmap = plt.get_cmap("tab20")
        for index, fragment in enumerate(fragment_list):
            colored_fragment = _copy_mesh(fragment)
            colored_fragment.paint_uniform_color(cmap(index % 20)[:3])
            geometries.append(colored_fragment)
    else:
        geometries.append(_mesh_with_normal_colors(assembled_mesh))

    screenshot_path = _ensure_output_dir(output_dir) / "reconstruction.png"
    return _show_geometries_non_blocking(
        geometries,
        title="Reconstruction",
        screenshot_path=screenshot_path,
        frames=60,
    )


def plot_match_matrix(
    scores_matrix: np.ndarray,
    fragment_names: Sequence[str],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a heatmap of pairwise fragment match scores."""
    output_path = _ensure_output_dir(output_dir) / "match_matrix.png"
    matrix = np.asarray(scores_matrix, dtype=float)

    fig_width = max(6, len(fragment_names) * 0.8)
    fig_height = max(5, len(fragment_names) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(fragment_names)))
    ax.set_yticks(range(len(fragment_names)))
    ax.set_xticklabels(fragment_names, rotation=45, ha="right")
    ax.set_yticklabels(fragment_names)
    ax.set_title("Pairwise Match Scores")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "--" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _extract_pair_rmse(metrics_dict: Dict[str, object]) -> Tuple[List[str], np.ndarray]:
    """Normalize pairwise RMSE metrics into labels and values."""
    pair_rmse = metrics_dict.get("pair_alignment_rmse", metrics_dict.get("pair_rmse", {}))

    if isinstance(pair_rmse, dict):
        labels = list(pair_rmse.keys())
        values = np.asarray(list(pair_rmse.values()), dtype=float) if pair_rmse else np.array([])
        return labels, values

    if isinstance(pair_rmse, list):
        labels = [f"pair_{idx}" for idx in range(len(pair_rmse))]
        return labels, np.asarray(pair_rmse, dtype=float)

    return [], np.array([], dtype=float)


def _extract_gap_distances(metrics_dict: Dict[str, object]) -> np.ndarray:
    """Normalize gap-distance metrics into a 1D array."""
    gap_distances = metrics_dict.get("gap_distances", [])
    return np.asarray(gap_distances, dtype=float).ravel()


def _extract_surface_breakdown(metrics_dict: Dict[str, object]) -> Dict[str, float]:
    """Normalize surface-classification counts."""
    breakdown = metrics_dict.get("surface_breakdown", {})
    if isinstance(breakdown, dict) and breakdown:
        return {str(key): float(value) for key, value in breakdown.items()}
    return {"break": 0.0, "original": 0.0, "uncertain": 0.0}


def plot_metrics_summary(
    metrics_dict: Dict[str, object],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save a compact summary figure for key reconstruction metrics."""
    output_path = _ensure_output_dir(output_dir) / "metrics_summary.png"
    pair_labels, pair_rmse = _extract_pair_rmse(metrics_dict)
    gap_distances = _extract_gap_distances(metrics_dict)
    surface_breakdown = _extract_surface_breakdown(metrics_dict)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if len(pair_rmse) > 0:
        axes[0].bar(pair_labels, pair_rmse, color="#4C72B0")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("RMSE")
    else:
        axes[0].text(0.5, 0.5, "No pair RMSE data", ha="center", va="center")
        axes[0].set_axis_off()
    axes[0].set_title("Per-Pair Alignment RMSE")

    if len(gap_distances) > 0:
        axes[1].hist(gap_distances, bins=20, color="#55A868", edgecolor="black")
        axes[1].set_xlabel("Gap distance")
        axes[1].set_ylabel("Count")
    else:
        axes[1].text(0.5, 0.5, "No gap data", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].set_title("Gap Distance Histogram")

    values = list(surface_breakdown.values())
    labels = list(surface_breakdown.keys())
    if sum(values) > 0:
        axes[2].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    else:
        axes[2].text(0.5, 0.5, "No surface breakdown", ha="center", va="center")
        axes[2].set_axis_off()
    axes[2].set_title("Surface Classification Breakdown")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def export_reconstruction_plotly(
    assembled_points: np.ndarray | o3d.geometry.PointCloud,
    colors: np.ndarray | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> go.Figure:
    """Export an interactive Plotly view of the reconstruction."""
    output_path = _ensure_output_dir(output_dir) / "reconstruction_interactive.html"

    if isinstance(assembled_points, o3d.geometry.PointCloud):
        points = np.asarray(assembled_points.points)
        point_colors = np.asarray(assembled_points.colors) if assembled_points.has_colors() else None
    else:
        points = np.asarray(assembled_points, dtype=float)
        point_colors = None

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("assembled_points must be an Nx3 array or an Open3D PointCloud")

    if colors is None:
        if point_colors is not None and len(point_colors) == len(points):
            marker_color = [
                f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in point_colors
            ]
        else:
            marker_color = _stone_palette_from_points(points)
    else:
        color_array = np.asarray(colors)
        if color_array.ndim == 2 and color_array.shape[1] == 3:
            marker_color = [
                f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in color_array
            ]
        else:
            marker_color = color_array

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "color": marker_color,
                    "opacity": 0.85,
                },
            )
        ]
    )
    figure.update_layout(
        title="Healing Stones Reconstruction",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    figure.write_html(str(output_path))
    return figure


def plot_match_scores(
    match_summary: List[Dict[str, float]],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Compatibility helper that saves a simple bar chart of top match scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "match_scores.png"

    if not match_summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No matches available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    labels = [f'{item["fragment_a"]}-{item["fragment_b"]}' for item in match_summary]
    scores = [item["score"] for item in match_summary]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, scores, color="#4C72B0")
    ax.set_title("Top Fragment Match Scores")
    ax.set_xlabel("Fragment Pair")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
