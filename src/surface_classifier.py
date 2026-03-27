"""Surface feature extraction and break-surface classification helpers.

Break surfaces are where the stone fractured, so they tend to be rougher and
less regular than the carved original surface. The classifier here is mainly a
search-space reduction step: isolate likely fracture regions first, then do the
expensive matching work on the most relevant geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import (
    CURVATURE_NEIGHBORS,
    LOGGER,
    MODELS_DIR,
    SURFACE_FEATURE_CHUNK_SIZE,
)


EPSILON = 1e-8
SURFACE_MODEL_PATH = Path(MODELS_DIR) / "surface_classifier.pkl"
LEGACY_SURFACE_MODEL_PATH = Path(MODELS_DIR) / "surface_classifier.joblib"
RF_CONFIDENCE_THRESHOLD = 0.8


@dataclass
class SurfaceClassificationResult:
    """Small summary object used by the current reconstruction pipeline."""

    break_score: float
    label: str
    metadata: Dict[str, float]


def _mesh_to_vertex_pointcloud(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    """Convert mesh vertices into a point cloud for faster surface processing."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    else:
        mesh.compute_vertex_normals()
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    return pcd


def _ensure_pointcloud_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Estimate normals on a point cloud when they are missing."""
    working_pcd = o3d.geometry.PointCloud(pcd)
    if working_pcd.is_empty():
        raise ValueError("Cannot classify an empty point cloud")

    if not working_pcd.has_normals() or len(working_pcd.normals) != len(working_pcd.points):
        bbox_diagonal = float(
            np.linalg.norm(working_pcd.get_axis_aligned_bounding_box().get_extent())
        )
        radius = max(bbox_diagonal * 0.03, 1e-3)
        working_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        working_pcd.normalize_normals()
    return working_pcd


def _query_nearest_indices(
    source_points: np.ndarray,
    reference_pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Find the nearest working-cloud point for every source point."""
    tree = o3d.geometry.KDTreeFlann(reference_pcd)
    indices = np.zeros(len(source_points), dtype=int)

    for point_index, point in enumerate(source_points):
        _, neighbor_indices, _ = tree.search_knn_vector_3d(point, 1)
        indices[point_index] = int(neighbor_indices[0]) if neighbor_indices else 0
    return indices


def transfer_labels_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project working-resolution labels back to the full-resolution mesh."""
    if mesh.is_empty() or len(mesh.vertices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if pcd.is_empty() or len(pcd.points) == 0:
        raise ValueError("Cannot project labels from an empty point cloud")

    if len(labels) != len(pcd.points):
        raise ValueError("labels must match the number of working point-cloud points")

    full_vertices = np.asarray(mesh.vertices)
    nearest_indices = _query_nearest_indices(full_vertices, pcd)
    transferred_labels = labels[nearest_indices].astype(int)

    if confidence is None:
        transferred_confidence = np.ones(len(transferred_labels), dtype=float)
    else:
        transferred_confidence = confidence[nearest_indices].astype(float)

    return transferred_labels, transferred_confidence


def compute_surface_features(
    mesh: o3d.geometry.TriangleMesh | None,
    pcd: o3d.geometry.PointCloud | None,
    k_neighbors: int = CURVATURE_NEIGHBORS,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Compute per-point geometric features used for surface classification.

    This runs only on the working-resolution point cloud so large fragments stay
    tractable. KDTree queries still happen point by point, but the heavy
    covariance math is batched in chunks afterward.
    """
    if pcd is None:
        if mesh is None:
            raise ValueError("Either mesh or pcd must be provided")
        working_pcd = _mesh_to_vertex_pointcloud(mesh)
    else:
        working_pcd = o3d.geometry.PointCloud(pcd)

    working_pcd = _ensure_pointcloud_normals(working_pcd)
    points = np.asarray(working_pcd.points)
    normals = np.asarray(working_pcd.normals)

    if len(points) == 0:
        raise ValueError("Cannot compute surface features for an empty point cloud")
    if len(points) < 3:
        raise ValueError("Need at least three points to estimate local surface features")

    if k_neighbors < 3:
        raise ValueError("k_neighbors must be at least 3")

    neighbor_count = min(max(k_neighbors, 3), len(points))
    feature_matrix = np.zeros((len(points), 5), dtype=float)
    neighbor_indices = np.zeros((len(points), neighbor_count), dtype=np.int32)
    valid_counts = np.full(len(points), neighbor_count, dtype=np.int32)
    tree = o3d.geometry.KDTreeFlann(working_pcd)

    query_iter = tqdm(
        range(len(points)),
        desc=progress_desc or "Surface neighborhoods",
        leave=False,
        disable=len(points) < 1000,
    )
    for point_index in query_iter:
        count, indices, _ = tree.search_knn_vector_3d(points[point_index], neighbor_count)
        count = max(int(count), 1)
        if count < neighbor_count:
            padded = np.full(neighbor_count, point_index, dtype=np.int32)
            padded[:count] = np.asarray(indices[:count], dtype=np.int32)
            neighbor_indices[point_index] = padded
        else:
            neighbor_indices[point_index] = np.asarray(indices[:neighbor_count], dtype=np.int32)
        valid_counts[point_index] = max(count, 3)

    chunk_size = SURFACE_FEATURE_CHUNK_SIZE
    chunk_iter = tqdm(
        range(0, len(points), chunk_size),
        desc=(progress_desc or "Surface features") + " (batched)",
        leave=False,
        disable=len(points) < 1000,
    )
    for start in chunk_iter:
        end = min(start + chunk_size, len(points))
        chunk_indices = np.arange(start, end)
        chunk_neighbors = neighbor_indices[start:end]
        chunk_counts = valid_counts[start:end].astype(float)

        local_points = points[chunk_neighbors]
        local_normals = normals[chunk_neighbors]
        reference_points = points[chunk_indices][:, None, :]
        centers = local_points.mean(axis=1, keepdims=True)
        centered = local_points - centers

        covariances = np.einsum("nki,nkj->nij", centered, centered)
        covariances /= chunk_counts[:, None, None]
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        plane_normals = eigenvectors[:, :, 0]

        signed_distances = np.einsum("nki,ni->nk", centered, plane_normals)
        radial_distances = np.linalg.norm(local_points - reference_points, axis=2)
        mean_radius = np.maximum(radial_distances.mean(axis=1), EPSILON)

        mean_curvature = np.mean(np.abs(signed_distances), axis=1) / mean_radius
        gaussian_curvature = (eigenvalues[:, 0] * eigenvalues[:, 1]) / (
            (eigenvalues.sum(axis=1) + EPSILON) ** 2
        )
        roughness = np.sqrt(np.mean(signed_distances**2, axis=1))

        average_normals = local_normals.mean(axis=1)
        average_normals /= np.maximum(
            np.linalg.norm(average_normals, axis=1, keepdims=True),
            EPSILON,
        )
        cosine_similarities = np.clip(
            np.einsum("nki,ni->nk", local_normals, average_normals),
            -1.0,
            1.0,
        )
        normal_variance = np.mean(1.0 - cosine_similarities, axis=1)
        height_variation = np.ptp(signed_distances, axis=1)

        feature_matrix[start:end] = np.column_stack(
            [
                mean_curvature,
                gaussian_curvature,
                roughness,
                normal_variance,
                height_variation,
            ]
        )

    return feature_matrix


def _pseudo_label_with_kmeans(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster features into rougher and smoother surface groups."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    if len(features) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if len(features) < 2 or np.allclose(features, features[0]):
        return np.zeros(len(features), dtype=int), np.ones(len(features), dtype=float)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(scaled_features)
    centers = kmeans.cluster_centers_

    roughness_score = centers[:, [0, 2, 3, 4]].mean(axis=1)
    break_cluster = int(np.argmax(roughness_score))
    labels = (cluster_ids == break_cluster).astype(int)

    distances = kmeans.transform(scaled_features)
    assigned_distances = distances[np.arange(len(cluster_ids)), cluster_ids]
    other_distances = distances[np.arange(len(cluster_ids)), 1 - cluster_ids]
    confidence = 1.0 - assigned_distances / (assigned_distances + other_distances + EPSILON)
    confidence = np.clip(confidence, 0.5, 1.0)

    return labels, confidence


def generate_pseudolabels(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-labels from geometric features with k-means.

    This is the first pass of the semi-supervised classifier: we start with an
    unsupervised guess, then let the Random Forest learn cross-fragment patterns
    from those labels.
    """
    return _pseudo_label_with_kmeans(features)


def generate_training_labels(
    mesh: o3d.geometry.TriangleMesh | None, features: np.ndarray
) -> np.ndarray:
    """Generate pseudo-labels from surface features using k-means clustering."""
    del mesh

    # TODO: Replace this pseudo-labeling step with a small manually annotated
    # training set. Even a modest amount of expert labeling would help a lot.
    labels, _ = generate_pseudolabels(features)
    return labels


def train_classifier_from_pseudolabels(
    all_features: np.ndarray,
    all_pseudolabels: np.ndarray,
) -> RandomForestClassifier:
    """Train a lightweight Random Forest from k-means pseudo-labels."""
    if len(all_features) != len(all_pseudolabels):
        raise ValueError("features and labels must have the same number of rows")

    unique_labels = np.unique(all_pseudolabels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least two classes to train the classifier")

    _, counts = np.unique(all_pseudolabels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("Each class needs at least two samples for a stratified split")

    if len(all_pseudolabels) < 5:
        raise ValueError("Not enough samples to train and validate the classifier")

    x_train, x_val, y_train, y_val = train_test_split(
        all_features,
        all_pseudolabels,
        test_size=0.2,
        random_state=42,
        stratify=all_pseudolabels,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    validation_metrics = {
        "accuracy": float(accuracy),
        "original_precision": float(
            precision_score(y_val, predictions, pos_label=0, zero_division=0)
        ),
        "original_recall": float(
            recall_score(y_val, predictions, pos_label=0, zero_division=0)
        ),
        "break_precision": float(
            precision_score(y_val, predictions, pos_label=1, zero_division=0)
        ),
        "break_recall": float(
            recall_score(y_val, predictions, pos_label=1, zero_division=0)
        ),
    }
    LOGGER.info("Surface classifier validation accuracy: %.4f", validation_metrics["accuracy"])
    LOGGER.info(
        "Original surface (0): precision=%.4f, recall=%.4f",
        validation_metrics["original_precision"],
        validation_metrics["original_recall"],
    )
    LOGGER.info(
        "Break surface (1): precision=%.4f, recall=%.4f",
        validation_metrics["break_precision"],
        validation_metrics["break_recall"],
    )
    model.validation_accuracy_ = validation_metrics["accuracy"]
    model.validation_metrics_ = validation_metrics

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SURFACE_MODEL_PATH)
    LOGGER.info("Saved semi-supervised surface classifier to %s", SURFACE_MODEL_PATH)
    return model


def train_surface_classifier(
    features: np.ndarray, labels: np.ndarray
) -> RandomForestClassifier:
    """Compatibility wrapper for older code paths."""
    return train_classifier_from_pseudolabels(features, labels)


def refine_with_confidence(
    model: RandomForestClassifier,
    features: np.ndarray,
    pseudolabels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend RF predictions with pseudo-labels using prediction confidence."""
    if len(features) != len(pseudolabels):
        raise ValueError("features and pseudolabels must have the same number of rows")

    if not hasattr(model, "predict_proba"):
        predictions = model.predict(features).astype(int)
        confidence = np.ones(len(predictions), dtype=float)
        return predictions, confidence

    probabilities = model.predict_proba(features)
    rf_predictions = probabilities.argmax(axis=1).astype(int)
    rf_confidence = probabilities.max(axis=1)

    use_rf_prediction = rf_confidence > RF_CONFIDENCE_THRESHOLD
    refined_labels = np.where(use_rf_prediction, rf_predictions, pseudolabels).astype(int)
    refined_confidence = np.where(
        use_rf_prediction,
        rf_confidence,
        np.clip(rf_confidence, 0.5, RF_CONFIDENCE_THRESHOLD),
    )
    return refined_labels, refined_confidence


def classify_surfaces(
    mesh: o3d.geometry.TriangleMesh | None,
    pcd: o3d.geometry.PointCloud | None,
    model: RandomForestClassifier | None = None,
    progress_desc: str | None = None,
    return_full_resolution: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify each working point as original surface or break surface."""
    if pcd is None:
        if mesh is None:
            raise ValueError("Either mesh or pcd must be provided")
        working_pcd = _mesh_to_vertex_pointcloud(mesh)
    else:
        working_pcd = o3d.geometry.PointCloud(pcd)

    features = compute_surface_features(
        mesh=None,
        pcd=working_pcd,
        progress_desc=progress_desc,
    )

    pseudolabels, pseudoconfidence = generate_pseudolabels(features)
    if model is not None:
        predicted_labels, predicted_confidence = refine_with_confidence(
            model,
            features,
            pseudolabels,
        )
    else:
        predicted_labels, predicted_confidence = pseudolabels, pseudoconfidence

    if return_full_resolution and mesh is not None:
        return transfer_labels_to_mesh(
            mesh,
            working_pcd,
            predicted_labels,
            predicted_confidence,
        )

    return predicted_labels, predicted_confidence


def extract_break_surface(
    mesh: o3d.geometry.TriangleMesh | None,
    pcd: o3d.geometry.PointCloud | None,
    labels: np.ndarray,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Extract the subset of points classified as break surface."""
    if pcd is not None:
        if len(labels) != len(pcd.points):
            raise ValueError("labels must match the number of working point-cloud points")

        break_indices = np.flatnonzero(labels == 1)
        break_cloud = o3d.geometry.PointCloud()
        if len(break_indices) == 0:
            return break_cloud, break_indices

        break_cloud.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points)[break_indices]
        )
        if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
            break_cloud.normals = o3d.utility.Vector3dVector(
                np.asarray(pcd.normals)[break_indices]
            )
        return break_cloud, break_indices

    if mesh is None:
        raise ValueError("Either mesh or pcd must be provided")

    if len(labels) != len(mesh.vertices):
        raise ValueError("labels must have the same length as the number of mesh vertices")

    break_indices = np.flatnonzero(labels == 1)
    break_cloud = o3d.geometry.PointCloud()

    if len(break_indices) == 0:
        return break_cloud, break_indices

    vertices = np.asarray(mesh.vertices)[break_indices]
    break_cloud.points = o3d.utility.Vector3dVector(vertices)

    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        normals = np.asarray(mesh.vertex_normals)[break_indices]
        break_cloud.normals = o3d.utility.Vector3dVector(normals)

    return break_cloud, break_indices


def classify_surface(mesh: o3d.geometry.TriangleMesh) -> SurfaceClassificationResult:
    """Compatibility wrapper that summarizes break-surface predictions per mesh."""
    labels, confidence = classify_surfaces(mesh=mesh, pcd=None, model=None)

    if len(labels) == 0:
        return SurfaceClassificationResult(
            break_score=0.0,
            label="unknown",
            metadata={"break_vertices": 0.0, "mean_confidence": 0.0},
        )

    break_score = float(labels.mean())
    label = "break_dominant" if break_score >= 0.5 else "original_dominant"

    return SurfaceClassificationResult(
        break_score=break_score,
        label=label,
        metadata={
            "break_vertices": float(labels.sum()),
            "total_vertices": float(len(labels)),
            "mean_confidence": float(confidence.mean()),
        },
    )
