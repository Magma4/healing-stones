"""Project configuration and shared constants."""

from __future__ import annotations

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fragments"
OUTPUT_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

VOXEL_SIZE = 0.5
FPFH_RADIUS = 2.5
ICP_THRESHOLD = 1.0
CURVATURE_NEIGHBORS = 30
RANSAC_DISTANCE = 1.5
ROUGHNESS_NEIGHBORS = 15
WORKING_MAX_VERTICES = 100000
FAST_MAX_VERTICES = 50000
SURFACE_CLASSIFIER_MAX_VERTICES = 120000
SURFACE_CLASSIFIER_START_VOXEL = 2.0
FEATURE_MAX_POINTS = 30000
MATCH_MAX_POINTS = 15000
LARGE_FRAGMENT_MIN_VOXEL = 2.0
FEATURE_VOXEL_DIAGONAL_DIVISOR = 200.0
SURFACE_FEATURE_CHUNK_SIZE = 4096


def setup_logger(name: str = "healing_stones") -> logging.Logger:
    """Create a simple console logger for the project."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logger()
