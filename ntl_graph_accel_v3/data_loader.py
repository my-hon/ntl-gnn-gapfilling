"""
Data loading and preprocessing for ntl_graph_accel_v3.

Handles:
  - Loading NTL TIF/NumPy data
  - Loading quality flags
  - Preprocessing: 65535->NaN, quality>1->NaN, /10.0, /100.0
  - Natural breaks sampling for training
  - NaN position extraction for prediction
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from ntl_graph_accel_v3.config import Config


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_data(path: str) -> np.ndarray:
    """Load NTL data from TIF or NumPy file.

    Attempts to load using rasterio (preferred for TIF), then GDAL,
    then falls back to NumPy (.npy) loading.

    Args:
        path: Path to the data file.

    Returns:
        3D numpy array (uint16 for raw TIF data).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        return _load_tif(path)
    elif ext == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _load_tif(path: str) -> np.ndarray:
    """Load a TIF file using rasterio or GDAL.

    Args:
        path: Path to the TIF file.

    Returns:
        3D numpy array.
    """
    # Try rasterio first
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()  # (bands, height, width) or (1, height, width)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]  # Remove single band dimension
            return data
    except ImportError:
        pass

    # Try GDAL
    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is None:
            raise ValueError(f"GDAL cannot open: {path}")
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        ds = None
        return data
    except ImportError:
        pass

    raise ImportError(
        "Cannot load TIF file: neither rasterio nor GDAL is available. "
        "Install with: pip install rasterio or pip install GDAL"
    )


def load_quality_flags(path: str) -> Optional[np.ndarray]:
    """Load quality flag data from TIF or NumPy file.

    Args:
        path: Path to the quality flag file. If empty, returns None.

    Returns:
        Quality flag array, or None if path is empty.
    """
    if not path:
        return None
    return load_data(path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    data: np.ndarray,
    quality: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply the reference preprocessing pipeline.

    Steps (matching reference exactly):
      1. Replace 65535 (NoData) with NaN
      2. Filter by quality flag (Mandatory_Quality_Flag > 1 -> set to NaN)
      3. Divide by 10.0 to get real radiance values (nW cm-2 sr-1)
      4. Divide by 100.0 for normalization (long-tail distribution, simple scaling)

    Args:
        data: Raw uint16 data array.
        quality: Optional quality flag array (same shape as data).

    Returns:
        Preprocessed float32 array with NaN for invalid pixels.
    """
    # Work with float32 from the start
    result = data.astype(np.float32)

    # Step 1: Replace 65535 with NaN
    result[result == 65535] = np.nan

    # Step 2: Filter by quality flag
    if quality is not None:
        quality_mask = quality > 1
        result[quality_mask] = np.nan

    # Step 3: Divide by 10.0 (real radiance)
    result = result / 10.0

    # Step 4: Divide by 100.0 (normalization)
    result = result / 100.0

    return result


# ---------------------------------------------------------------------------
# Natural breaks sampling for training
# ---------------------------------------------------------------------------

def natural_breaks_sampling(
    data: np.ndarray,
    config: Config,
) -> List[Tuple[int, int, int]]:
    """Sample positions using natural breaks stratification.

    Divides valid pixel values into categories based on natural breaks,
    then samples `sample_per_class` positions from each category.

    Valid region: data[EDGE_TIME:-EDGE_TIME, EDGE_HEIGHT:-EDGE_HEIGHT,
                       EDGE_WIDTH:-EDGE_WIDTH]

    Args:
        data: Preprocessed float32 data array (with NaN for invalid).
        config: Configuration object.

    Returns:
        List of (t, h, w) position tuples for graph construction.
    """
    breaks = config.natural_breaks
    n_categories = config.num_categories
    sample_per_class = config.sample_per_class

    # Extract valid region
    t_slice, h_slice, w_slice = config.valid_slice
    valid_data = data[t_slice, h_slice, w_slice]

    # Get valid (non-NaN) positions and values
    valid_mask = ~np.isnan(valid_data)
    valid_positions = np.argwhere(valid_mask)  # (N, 3) array of (t, h, w)
    valid_values = valid_data[valid_mask]

    if len(valid_values) == 0:
        return []

    # Digitize into categories
    categories = np.digitize(valid_values, breaks) - 1
    # Clip to valid range [0, n_categories-1]
    categories = np.clip(categories, 0, n_categories - 1)

    # Sample from each category
    rng = np.random.RandomState(config.seed)
    sampled_positions = []

    for cat in range(n_categories):
        cat_mask = categories == cat
        cat_indices = np.where(cat_mask)[0]

        if len(cat_indices) == 0:
            continue

        # Sample up to sample_per_class
        n_sample = min(sample_per_class, len(cat_indices))
        sampled_idx = rng.choice(cat_indices, size=n_sample, replace=False)

        for idx in sampled_idx:
            # Convert local position to global position
            local_t, local_h, local_w = valid_positions[idx]
            global_t = local_t + config.edge_time
            global_h = local_h + config.edge_height
            global_w = local_w + config.edge_width
            sampled_positions.append((int(global_t), int(global_h), int(global_w)))

    return sampled_positions


# ---------------------------------------------------------------------------
# NaN position extraction for prediction
# ---------------------------------------------------------------------------

def extract_nan_positions_by_day(
    data: np.ndarray,
    config: Config,
) -> List[Tuple[int, List[Tuple[int, int, int]]]]:
    """Extract NaN positions day by day within the valid region.

    For prediction mode: process each day (time slice) independently,
    finding all NaN positions within the valid spatial region.

    Args:
        data: Preprocessed float32 data array (with NaN for invalid).
        config: Configuration object.

    Returns:
        List of (day_index, [(h, w), ...]) tuples, one per day that has
        at least one NaN position in the valid region.
    """
    t_slice, h_slice, w_slice = config.valid_slice
    valid_data = data[t_slice, h_slice, w_slice]

    t_start = config.edge_time
    t_end = config.data_shape[0] - config.edge_time
    h_start = config.edge_height
    h_end = config.data_shape[1] - config.edge_height
    w_start = config.edge_width
    w_end = config.data_shape[2] - config.edge_width

    result = []
    for day in range(t_start, t_end):
        day_data = data[day, h_start:h_end, w_start:w_end]
        nan_mask = np.isnan(day_data)
        nan_positions = np.argwhere(nan_mask)  # (N, 2) array of (h, w)

        if len(nan_positions) > 0:
            positions = [
                (int(h_start + pos[0]), int(w_start + pos[1]))
                for pos in nan_positions
            ]
            result.append((day, positions))

    return result


# ---------------------------------------------------------------------------
# DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    """High-level data loader with preprocessing and sampling.

    Args:
        config: Configuration object.
    """

    def __init__(self, config: Config):
        self.config = config
        self._data: Optional[np.ndarray] = None
        self._quality: Optional[np.ndarray] = None
        self._preprocessed: Optional[np.ndarray] = None

    def load_raw(self) -> np.ndarray:
        """Load raw data from disk.

        Returns:
            Raw data array (uint16 or original dtype).
        """
        if self._data is None:
            self._data = load_data(self.config.input_path)
        return self._data

    def load_quality(self) -> Optional[np.ndarray]:
        """Load quality flags from disk.

        Returns:
            Quality flag array, or None if no quality path is set.
        """
        if self._quality is None and self.config.quality_path:
            self._quality = load_quality_flags(self.config.quality_path)
        return self._quality

    def get_preprocessed(self) -> np.ndarray:
        """Load and preprocess data.

        Returns:
            Preprocessed float32 array with NaN for invalid pixels.
        """
        if self._preprocessed is None:
            raw = self.load_raw()
            quality = self.load_quality()
            self._preprocessed = preprocess(raw, quality)
        return self._preprocessed

    def get_training_positions(self) -> List[Tuple[int, int, int]]:
        """Get stratified training sample positions.

        Returns:
            List of (t, h, w) tuples sampled via natural breaks.
        """
        data = self.get_preprocessed()
        return natural_breaks_sampling(data, self.config)

    def get_prediction_positions(
        self,
    ) -> List[Tuple[int, List[Tuple[int, int, int]]]]:
        """Get NaN positions for prediction, organized by day.

        Returns:
            List of (day_index, [(h, w), ...]) tuples.
        """
        data = self.get_preprocessed()
        return extract_nan_positions_by_day(data, self.config)

    def save_preprocessed(self, path: Optional[str] = None) -> None:
        """Save preprocessed data to disk.

        Args:
            path: Output path. If None, uses cache_dir/preprocessed.npy.
        """
        if path is None:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            path = os.path.join(self.config.cache_dir, "preprocessed.npy")
        data = self.get_preprocessed()
        np.save(path, data)
