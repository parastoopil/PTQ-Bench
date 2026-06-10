"""Shared pytest fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_image() -> np.ndarray:
    """640×640 random BGR image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)


@pytest.fixture
def batch_images() -> list:
    return [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]
