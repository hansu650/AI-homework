"""Lightweight NYUDepthV2 MAT-file IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DATASETS = ("images", "depths", "labels")


def find_nyu_labeled_mat(data_root: Path) -> Path:
    """Find nyu_depth_v2_labeled with or without a .mat extension."""

    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise ValueError(f"Data root is not a directory: {data_root}")

    candidates = [
        data_root / "nyu_depth_v2_labeled.mat",
        data_root / "nyu_depth_v2_labeled",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find nyu_depth_v2_labeled or nyu_depth_v2_labeled.mat "
        f"under {data_root}"
    )


def inspect_mat_file(mat_path: Path) -> dict[str, Any]:
    """Inspect HDF5 keys, shapes, and dtypes without reading large arrays."""

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    if not mat_path.is_file():
        raise ValueError(f"MAT path is not a file: {mat_path}")

    try:
        h5_file = h5py.File(mat_path, "r")
    except OSError as error:
        return {
            "path": str(mat_path),
            "is_hdf5": False,
            "error": str(error),
            "message": "File is not readable by h5py; it may be a pre-v7.3 MAT file.",
        }

    with h5_file as handle:
        datasets: dict[str, dict[str, str | tuple[int, ...]]] = {}
        groups: list[str] = []
        for key in sorted(handle.keys()):
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                datasets[key] = {
                    "shape": tuple(int(dim) for dim in obj.shape),
                    "dtype": str(obj.dtype),
                }
            else:
                groups.append(key)

        for required in REQUIRED_DATASETS:
            if required not in datasets:
                raise KeyError(f"Required dataset '{required}' missing in {mat_path}")

        return {
            "path": str(mat_path),
            "is_hdf5": True,
            "keys": sorted(handle.keys()),
            "datasets": datasets,
            "groups": groups,
            "sample_count": int(handle["images"].shape[0]),
            "has_names": "names" in datasets,
        }


def read_nyu_mat_sample(mat_path: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one sample from a v7.3 NYUDepthV2 MAT file by index."""

    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as handle:
        _validate_required_datasets(handle, mat_path)
        sample_count = int(handle["images"].shape[0])
        if index < 0 or index >= sample_count:
            raise IndexError(f"Sample index {index} out of range 0..{sample_count - 1}")

        depth = _format_hw_array(handle["depths"][index], "depth")
        label = _format_hw_array(handle["labels"][index], "label").astype(np.int64)
        rgb = _format_rgb_array(handle["images"][index], expected_hw=depth.shape)

    if label.shape != depth.shape or rgb.shape[:2] != depth.shape:
        raise ValueError(
            "RGB, depth, and label shapes do not match after formatting: "
            f"rgb={rgb.shape}, depth={depth.shape}, label={label.shape}"
        )
    return rgb, depth.astype(np.float32), label


def normalize_depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    """Normalize a 2D float depth map into a 16-bit PNG-friendly array."""

    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("depth contains NaN or infinite values")

    depth = depth.astype(np.float32)
    depth = depth - float(depth.min())
    max_value = float(depth.max())
    if max_value > 0.0:
        depth = depth / max_value
    return np.round(depth * 65535.0).astype(np.uint16)


def read_nyu_names(mat_path: Path) -> list[str]:
    """Read the small NYU names table from the MAT file."""

    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as handle:
        if "names" not in handle:
            raise KeyError(
                "Dataset 'names' missing; automatic keyword mapping needs manual "
                "label-id confirmation."
            )
        names_dataset = handle["names"]
        if names_dataset.ndim != 2:
            raise ValueError(f"names dataset must be 2D, got shape {names_dataset.shape}")

        names: list[str] = []
        for index in range(names_dataset.shape[1]):
            ref = names_dataset[0, index]
            names.append(_read_matlab_string(handle, ref))
        return names


def _validate_required_datasets(handle: h5py.File, mat_path: Path) -> None:
    for key in REQUIRED_DATASETS:
        if key not in handle:
            raise KeyError(f"Required dataset '{key}' missing in {mat_path}")


def _format_hw_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be HxW, got shape {array.shape}")
    return array


def _format_rgb_array(array: np.ndarray, expected_hw: tuple[int, int]) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"RGB array must be 3D, got shape {array.shape}")

    if array.shape[0] == 3:
        rgb = np.transpose(array, (1, 2, 0))
    elif array.shape[2] == 3 and array.shape[:2] == expected_hw:
        rgb = array
    elif array.shape[2] == 3 and array.shape[:2] == expected_hw[::-1]:
        rgb = np.transpose(array, (1, 0, 2))
    else:
        raise ValueError(
            f"Unsupported RGB shape {array.shape}; expected CxHxW or HxWxC "
            f"matching depth shape {expected_hw}"
        )

    if rgb.shape[:2] != expected_hw or rgb.shape[2] != 3:
        raise ValueError(
            f"RGB shape after formatting is {rgb.shape}, expected {expected_hw}x3"
        )
    return rgb.astype(np.uint8)


def _read_matlab_string(handle: h5py.File, ref: h5py.Reference) -> str:
    if not ref:
        raise ValueError("Encountered an empty MATLAB string reference")
    chars = np.asarray(handle[ref][()]).reshape(-1)
    return "".join(chr(int(value)) for value in chars if int(value) != 0)
