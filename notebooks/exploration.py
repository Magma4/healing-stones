# %% [markdown]
# # Healing Stones Exploration
#
# This notebook is an early exploratory pass over the 3D fragment dataset for the
# Healing Stones reconstruction task. The goal here is not to prove the whole
# pipeline works perfectly, but to inspect the data, test the main geometric
# signals, and document what seems promising and what still looks fragile.
#
# I wrote this in percent format so it works in both VS Code and Jupyter.

# %%
"""Exploratory notebook-script for early Healing Stones experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import compute_mesh_stats, load_fragment, mesh_to_pointcloud, preprocess_mesh
from src.feature_extraction import compute_fpfh_features
from src.matching import compute_match_score, match_fpfh
from src.surface_classifier import (
    classify_surfaces,
    compute_surface_features,
    extract_break_surface,
)


DATA_DIR = PROJECT_ROOT / "data" / "fragments"
fragment_files = sorted(
    path for path in DATA_DIR.glob("*") if path.suffix.lower() in {".ply", ".obj"}
)

assert fragment_files, f"No fragment files found in {DATA_DIR}"
fragment_files[:5]


# %%
def load_preprocessed_mesh(
    path: Path,
    voxel_size: float | None = None,
) -> o3d.geometry.TriangleMesh:
    mesh = load_fragment(path)
    return preprocess_mesh(mesh, voxel_size=voxel_size)


def mesh_figure(
    mesh: o3d.geometry.TriangleMesh,
    title: str,
    intensity: np.ndarray | None = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    kwargs = {}
    if intensity is not None:
        kwargs["intensity"] = np.asarray(intensity, dtype=float)
        kwargs["colorscale"] = colorscale
        kwargs["showscale"] = True
    else:
        kwargs["color"] = "lightgray"
        kwargs["opacity"] = 1.0

    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                flatshading=True,
                **kwargs,
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return figure


def pointcloud_figure(
    pcd: o3d.geometry.PointCloud,
    title: str,
    color: str | np.ndarray = "#4C72B0",
    size: int = 2,
) -> go.Figure:
    points = np.asarray(pcd.points)
    if isinstance(color, np.ndarray) and color.ndim == 2 and color.shape[1] == 3:
        marker_color = [
            f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in color
        ]
    else:
        marker_color = color

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": size, "color": marker_color, "opacity": 0.85},
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return figure


def vertex_scatter_figure(
    mesh: o3d.geometry.TriangleMesh,
    title: str,
    colors: np.ndarray,
    size: int = 2,
) -> go.Figure:
    vertices = np.asarray(mesh.vertices)
    marker_color = [
        f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in np.asarray(colors)
    ]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker={"size": size, "color": marker_color, "opacity": 0.9},
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return figure


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    value_min = values.min()
    value_max = values.max()
    if abs(value_max - value_min) < 1e-8:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def compare_alignment_figure(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    transform: np.ndarray,
    title: str,
) -> go.Figure:
    source_before = mesh_to_pointcloud(mesh_a, num_points=5000)
    target_before = mesh_to_pointcloud(mesh_b, num_points=5000)

    mesh_a_aligned = o3d.geometry.TriangleMesh(mesh_a)
    mesh_a_aligned.transform(transform)
    source_after = mesh_to_pointcloud(mesh_a_aligned, num_points=5000)
    target_after = mesh_to_pointcloud(mesh_b, num_points=5000)

    points_before_a = np.asarray(source_before.points)
    points_before_b = np.asarray(target_before.points)
    points_after_a = np.asarray(source_after.points)
    points_after_b = np.asarray(target_after.points)

    offset = max(
        float(np.linalg.norm(mesh_a.get_axis_aligned_bounding_box().get_extent())),
        float(np.linalg.norm(mesh_b.get_axis_aligned_bounding_box().get_extent())),
        1.0,
    ) * 1.8

    points_before_a = points_before_a - np.array([offset, 0.0, 0.0])
    points_before_b = points_before_b + np.array([offset, 0.0, 0.0])
    points_after_a = points_after_a + np.array([3.0 * offset, 0.0, 0.0])
    points_after_b = points_after_b + np.array([3.0 * offset, 0.0, 0.0])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=points_before_a[:, 0],
            y=points_before_a[:, 1],
            z=points_before_a[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#4C72B0"},
            name="Fragment A (before)",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=points_before_b[:, 0],
            y=points_before_b[:, 1],
            z=points_before_b[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#DD8452"},
            name="Fragment B (before)",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=points_after_a[:, 0],
            y=points_after_a[:, 1],
            z=points_after_a[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#4C72B0"},
            name="Fragment A (aligned)",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=points_after_b[:, 0],
            y=points_after_b[:, 1],
            z=points_after_b[:, 2],
            mode="markers",
            marker={"size": 2, "color": "#DD8452"},
            name="Fragment B (aligned)",
        )
    )

    # Draw only a small number of correspondence lines so the plot stays readable.
    target_pcd_after = o3d.geometry.PointCloud()
    target_pcd_after.points = o3d.utility.Vector3dVector(points_after_b)
    target_tree = o3d.geometry.KDTreeFlann(target_pcd_after)
    sample_indices = np.linspace(0, len(points_after_a) - 1, min(100, len(points_after_a)), dtype=int)

    for sample_index in sample_indices:
        source_point = points_after_a[sample_index]
        _, neighbor_indices, _ = target_tree.search_knn_vector_3d(source_point, 1)
        if not neighbor_indices:
            continue
        target_point = points_after_b[neighbor_indices[0]]
        figure.add_trace(
            go.Scatter3d(
                x=[source_point[0], target_point[0]],
                y=[source_point[1], target_point[1]],
                z=[source_point[2], target_point[2]],
                mode="lines",
                line={"color": "rgba(80,80,80,0.25)", "width": 2},
                showlegend=False,
            )
        )

    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return figure


def summarize_match(reg_result, source_pcd, target_pcd) -> dict[str, float]:
    score = compute_match_score(reg_result, source_pcd, target_pcd)
    return {
        "fitness": float(reg_result.fitness),
        "inlier_rmse": float(reg_result.inlier_rmse),
        "score": float(score),
        "num_correspondences": int(len(reg_result.correspondence_set)),
    }


# %% [markdown]
# ## 1. Load one fragment, print stats, visualize it interactively
#
# First I want to inspect a single fragment as a sanity check. At this stage I am
# mainly asking:
#
# - Is the mesh loaded cleanly?
# - Does the scale look consistent?
# - Do I see obvious break faces and obvious carved/original faces?

# %%
fragment_path = fragment_files[0]
fragment_mesh = load_preprocessed_mesh(fragment_path)
fragment_stats = compute_mesh_stats(fragment_mesh)

print("Fragment:", fragment_path.name)
print(fragment_stats)

mesh_figure(fragment_mesh, title=f"Fragment Preview: {fragment_path.name}")


# %% [markdown]
# ## 2. Curvature map on a single fragment
#
# Here I use the per-vertex surface features from `surface_classifier.py` and
# color the mesh by mean curvature. This is useful for checking whether the
# fracture face stands out as a rougher, less regular region.
#
# In practice I expect this to be helpful, but not enough on its own. Some carved
# areas can also have high curvature, and some break surfaces can be relatively
# smooth.

# %%
surface_features = compute_surface_features(fragment_mesh, pcd=None)
mean_curvature = surface_features[:, 0]

print("Curvature stats")
print(
    {
        "min": float(mean_curvature.min()),
        "mean": float(mean_curvature.mean()),
        "max": float(mean_curvature.max()),
    }
)

mesh_figure(
    fragment_mesh,
    title=f"Mean Curvature Map: {fragment_path.name}",
    intensity=normalize(mean_curvature),
    colorscale="Viridis",
)


# %% [markdown]
# ## 3. Roughness map
#
# Roughness is often more directly useful for this problem than curvature. A
# break surface usually has more local variation than the original carved face,
# even when its overall shape is fairly flat.
#
# This is one of the first things I would check when deciding whether the
# heuristic surface classifier is on the right track.

# %%
roughness = surface_features[:, 2]

print("Roughness stats")
print(
    {
        "min": float(roughness.min()),
        "mean": float(roughness.mean()),
        "max": float(roughness.max()),
    }
)

mesh_figure(
    fragment_mesh,
    title=f"Roughness Map: {fragment_path.name}",
    intensity=normalize(roughness),
    colorscale="Plasma",
)


# %% [markdown]
# ## 4. Surface classification on one fragment
#
# This uses the pseudo-labeled surface classifier. Right now it is still a
# heuristic / weakly supervised step, so I treat it as a practical first pass
# rather than a final result.
#
# Break surface vertices are shown in red, original surface vertices in blue, and
# low-confidence regions in yellow.

# %%
labels, confidence = classify_surfaces(fragment_mesh, pcd=None, model=None)
break_surface_pcd, break_vertex_indices = extract_break_surface(
    fragment_mesh,
    pcd=None,
    labels=labels,
)

classification_colors = np.zeros((len(labels), 3), dtype=float)
classification_colors[labels == 0] = np.array([0.2, 0.4, 0.95])   # original
classification_colors[labels == 1] = np.array([0.9, 0.15, 0.15])  # break
classification_colors[confidence < 0.6] = np.array([0.95, 0.8, 0.1])

print(
    {
        "break_vertices": int(np.sum(labels == 1)),
        "original_vertices": int(np.sum(labels == 0)),
        "uncertain_vertices": int(np.sum(confidence < 0.6)),
        "mean_confidence": float(confidence.mean()),
    }
)

vertex_scatter_figure(
    fragment_mesh,
    title=f"Surface Classification: {fragment_path.name}",
    colors=classification_colors,
    size=2,
)

pointcloud_figure(
    break_surface_pcd,
    title=f"Extracted Break Surface: {fragment_path.name}",
    color="#D62728",
    size=2,
)


# %% [markdown]
# ## 5. Compare two fragments that might fit
#
# For this step I compute break-surface point clouds, downsample them, build FPFH
# features, and run the RANSAC matcher. The RANSAC threshold in the project code
# is tied to the voxel size because a correspondence radius around 1-2x voxel
# size is a common practical choice.
#
# I am not assuming these two fragments definitely fit. The point here is to see
# what the coarse alignment looks like and how strong the score is.

# %%
fragment_a_path = fragment_files[0]
fragment_b_path = fragment_files[1]

fragment_a = load_preprocessed_mesh(fragment_a_path)
fragment_b = load_preprocessed_mesh(fragment_b_path)

labels_a, _ = classify_surfaces(fragment_a, pcd=None, model=None)
labels_b, _ = classify_surfaces(fragment_b, pcd=None, model=None)
break_a, _ = extract_break_surface(fragment_a, pcd=None, labels=labels_a)
break_b, _ = extract_break_surface(fragment_b, pcd=None, labels=labels_b)

down_a, fpfh_a = compute_fpfh_features(break_a, voxel_size=0.5)
down_b, fpfh_b = compute_fpfh_features(break_b, voxel_size=0.5)

fit_candidate = match_fpfh(down_a, down_b, fpfh_a, fpfh_b, voxel_size=0.5)
fit_candidate_summary = summarize_match(fit_candidate, break_a, break_b)

print("Pair 1:", fragment_a_path.name, "vs", fragment_b_path.name)
print(fit_candidate_summary)

compare_alignment_figure(
    fragment_a,
    fragment_b,
    fit_candidate.transformation,
    title=f"Candidate Match: {fragment_a_path.name} vs {fragment_b_path.name}",
)


# %% [markdown]
# ## 6. Try a pair that probably does **not** fit
#
# Here I intentionally compare fragments that are far apart in the sorted file
# list. This is not a guarantee they do not fit, but it is a reasonable negative
# check for early experimentation.
#
# What I want from this step is not "perfect rejection", but at least some
# separation in the final score compared with better candidate pairs.

# %%
fragment_bad_a_path = fragment_files[0]
fragment_bad_b_path = fragment_files[-1]

fragment_bad_a = load_preprocessed_mesh(fragment_bad_a_path)
fragment_bad_b = load_preprocessed_mesh(fragment_bad_b_path)

labels_bad_a, _ = classify_surfaces(fragment_bad_a, pcd=None, model=None)
labels_bad_b, _ = classify_surfaces(fragment_bad_b, pcd=None, model=None)
break_bad_a, _ = extract_break_surface(fragment_bad_a, pcd=None, labels=labels_bad_a)
break_bad_b, _ = extract_break_surface(fragment_bad_b, pcd=None, labels=labels_bad_b)

down_bad_a, fpfh_bad_a = compute_fpfh_features(break_bad_a, voxel_size=0.5)
down_bad_b, fpfh_bad_b = compute_fpfh_features(break_bad_b, voxel_size=0.5)

non_fit_candidate = match_fpfh(down_bad_a, down_bad_b, fpfh_bad_a, fpfh_bad_b, voxel_size=0.5)
non_fit_summary = summarize_match(non_fit_candidate, break_bad_a, break_bad_b)

print("Pair 2:", fragment_bad_a_path.name, "vs", fragment_bad_b_path.name)
print(non_fit_summary)

compare_alignment_figure(
    fragment_bad_a,
    fragment_bad_b,
    non_fit_candidate.transformation,
    title=f"Likely Non-Match: {fragment_bad_a_path.name} vs {fragment_bad_b_path.name}",
)


# %% [markdown]
# ## 7. Summary observations and notes
#
# This notebook is meant to show the reasoning process, so I want to be explicit
# about what I trust and what I do not trust yet.
#
# Current takeaways:
#
# - Curvature is useful for spotting locally complex areas, but it is not enough
#   by itself to separate break surfaces from carved/original surfaces.
# - Roughness looks more directly relevant for the fracture-vs-original question.
# - The pseudo-labeled surface classifier is a practical shortcut, not a final
#   answer. I expect it to work on clearly rough breaks and struggle on smoother
#   or weathered breaks.
# - FPFH + RANSAC is a reasonable first-pass matcher. It gives a geometric
#   alignment signal that can be compared across candidate pairs.
# - Negative checks are important. Even when two fragments do not belong
#   together, local geometry can sometimes still give a plausible coarse match.
#
# Honest limitations:
#
# - The curvature-based classification may look good on some fragments and fail
#   on others where the break surface is relatively smooth.
# - Matching quality will likely drop when there is substantial material loss,
#   because the true break surfaces may no longer be complementary.
# - The current notebook uses heuristic break-surface labels. A small manually
#   annotated training set would make this much more convincing.
#
# After running this notebook on more fragments, the next thing I would do is:
#
# 1. Log which fragments consistently produce clean break-surface masks.
# 2. Compare "good match" and "bad match" score distributions over many pairs.
# 3. Decide whether to trust the current classifier enough for full-pipeline
#    matching, or whether to fall back to using larger candidate surface regions.
