"""
Template precomputation for ntl_graph_accel_v3.

Precomputes distance matrices, offset matrices, and quadrant labels for a fixed
cube size. These templates are reused across all graph constructions with the
same cube dimensions, avoiding redundant computation.
"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _assign_quadrants_numpy(cube_shape: Tuple[int, int, int]) -> np.ndarray:
    """Assign quadrant labels using NumPy (fallback when Numba is unavailable).

    Plan A quadrant assignment:
      - |dt| largest, dt < 0 -> quadrant 1
      - |dt| largest, dt > 0 -> quadrant 2
      - |dh| largest, dh < 0 -> quadrant 3
      - |dh| largest, dh > 0 -> quadrant 4
      - |dw| largest, dw < 0 -> quadrant 5
      - |dw| largest, dw > 0 -> quadrant 6
      - center (0,0,0) -> 0 (not assigned to any quadrant)

    Args:
        cube_shape: Shape of the sub-cube (dt, dh, dw).

    Returns:
        ndarray of quadrant labels with shape cube_shape, dtype int8.
    """
    dt_size, dh_size, dw_size = cube_shape
    center_t = dt_size // 2
    center_h = dh_size // 2
    center_w = dw_size // 2

    # Build coordinate grids
    t_coords = np.arange(dt_size) - center_t
    h_coords = np.arange(dh_size) - center_h
    w_coords = np.arange(dw_size) - center_w

    dt = t_coords[:, None, None] * np.ones((1, dh_size, dw_size))
    dh = np.ones((dt_size, 1, 1)) * h_coords[None, :, None] * np.ones((1, 1, dw_size))
    dw = np.ones((dt_size, 1, 1)) * np.ones((1, dh_size, 1)) * w_coords[None, None, :]

    abs_dt = np.abs(dt)
    abs_dh = np.abs(dh)
    abs_dw = np.abs(dw)

    # Initialize with 0 (center)
    quads = np.zeros(cube_shape, dtype=np.int8)

    # Determine which axis has the largest absolute offset
    # For ties, NumPy's argmax picks the first (t > h > w)
    stacked = np.stack([abs_dt, abs_dh, abs_dw], axis=-1)  # (..., 3)
    max_axis = np.argmax(stacked, axis=-1)  # 0=t, 1=h, 2=w

    # Quadrant assignment based on max axis and sign
    # dt largest, dt < 0 -> 1
    mask = (max_axis == 0) & (dt < 0)
    quads[mask] = 1
    # dt largest, dt > 0 -> 2
    mask = (max_axis == 0) & (dt > 0)
    quads[mask] = 2
    # dh largest, dh < 0 -> 3
    mask = (max_axis == 1) & (dh < 0)
    quads[mask] = 3
    # dh largest, dh > 0 -> 4
    mask = (max_axis == 1) & (dh > 0)
    quads[mask] = 4
    # dw largest, dw < 0 -> 5
    mask = (max_axis == 2) & (dw < 0)
    quads[mask] = 5
    # dw largest, dw > 0 -> 6
    mask = (max_axis == 2) & (dw > 0)
    quads[mask] = 6

    return quads


def _compute_distances_offsets_numpy(
    cube_shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distances and offsets from center using NumPy.

    Args:
        cube_shape: Shape of the sub-cube (dt, dh, dw).

    Returns:
        distances: ndarray of shape cube_shape, dtype float32.
        offsets: ndarray of shape cube_shape + (3,), dtype float32.
    """
    dt_size, dh_size, dw_size = cube_shape
    center_t = dt_size // 2
    center_h = dh_size // 2
    center_w = dw_size // 2

    t_coords = np.arange(dt_size) - center_t
    h_coords = np.arange(dh_size) - center_h
    w_coords = np.arange(dw_size) - center_w

    dt = t_coords[:, None, None] * np.ones((1, dh_size, dw_size))
    dh = np.ones((dt_size, 1, 1)) * h_coords[None, :, None] * np.ones((1, 1, dw_size))
    dw = np.ones((dt_size, 1, 1)) * np.ones((1, dh_size, 1)) * w_coords[None, None, :]

    distances = np.sqrt(dt ** 2 + dh ** 2 + dw ** 2).astype(np.float32)
    offsets = np.stack([dt, dh, dw], axis=-1).astype(np.float32)

    return distances, offsets


if HAS_NUMBA:

    @njit(cache=True)
    def _assign_quadrants_numba(
        quads: np.ndarray,
        dt_size: int, dh_size: int, dw_size: int,
        center_t: int, center_h: int, center_w: int,
    ) -> None:
        """Numba-accelerated quadrant assignment (in-place)."""
        for t in range(dt_size):
            dt = t - center_t
            abs_dt = abs(dt)
            for h in range(dh_size):
                dh = h - center_h
                abs_dh = abs(dh)
                for w in range(dw_size):
                    dw = w - center_w
                    abs_dw = abs(dw)

                    if dt == 0 and dh == 0 and dw == 0:
                        quads[t, h, w] = 0
                        continue

                    # Find axis with largest absolute offset
                    if abs_dt >= abs_dh and abs_dt >= abs_dw:
                        if dt < 0:
                            quads[t, h, w] = 1
                        else:
                            quads[t, h, w] = 2
                    elif abs_dh >= abs_dt and abs_dh >= abs_dw:
                        if dh < 0:
                            quads[t, h, w] = 3
                        else:
                            quads[t, h, w] = 4
                    else:
                        if dw < 0:
                            quads[t, h, w] = 5
                        else:
                            quads[t, h, w] = 6

    @njit(cache=True)
    def _compute_distances_offsets_numba(
        distances: np.ndarray,
        offsets: np.ndarray,
        dt_size: int, dh_size: int, dw_size: int,
        center_t: int, center_h: int, center_w: int,
    ) -> None:
        """Numba-accelerated distance and offset computation (in-place)."""
        for t in range(dt_size):
            dt = t - center_t
            for h in range(dh_size):
                dh = h - center_h
                for w in range(dw_size):
                    dw = w - center_w
                    distances[t, h, w] = np.sqrt(
                        float(dt * dt + dh * dh + dw * dw)
                    )
                    offsets[t, h, w, 0] = float(dt)
                    offsets[t, h, w, 1] = float(dh)
                    offsets[t, h, w, 2] = float(dw)


class TemplateManager:
    """Manages precomputed template matrices for graph construction.

    Caches templates by cube shape to avoid recomputation. Templates include:
    - Quadrant labels (int8)
    - Euclidean distances from center (float32)
    - Offset vectors from center (float32, shape + (3,))
    - Sorted position indices by distance (for quick iteration)
    """

    def __init__(self):
        self._cache: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {}

    def get_templates(
        self, cube_shape: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        """Get or compute templates for the given cube shape.

        Args:
            cube_shape: Shape of the sub-cube (dt, dh, dw).

        Returns:
            Dictionary with keys:
                - 'quadrants': int8 array of quadrant labels
                - 'distances': float32 array of Euclidean distances
                - 'offsets': float32 array of offset vectors (..., 3)
                - 'sorted_indices': int32 array of positions sorted by distance
        """
        if cube_shape in self._cache:
            return self._cache[cube_shape]

        templates = self._compute(cube_shape)
        self._cache[cube_shape] = templates
        return templates

    def _compute(
        self, cube_shape: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        """Compute all templates for the given cube shape."""
        dt_size, dh_size, dw_size = cube_shape
        center_t = dt_size // 2
        center_h = dh_size // 2
        center_w = dw_size // 2

        # Compute quadrants
        if HAS_NUMBA:
            quads = np.zeros(cube_shape, dtype=np.int8)
            _assign_quadrants_numba(
                quads, dt_size, dh_size, dw_size,
                center_t, center_h, center_w,
            )
        else:
            quads = _assign_quadrants_numpy(cube_shape)

        # Compute distances and offsets
        distances = np.zeros(cube_shape, dtype=np.float32)
        offsets = np.zeros(cube_shape + (3,), dtype=np.float32)

        if HAS_NUMBA:
            _compute_distances_offsets_numba(
                distances, offsets,
                dt_size, dh_size, dw_size,
                center_t, center_h, center_w,
            )
        else:
            distances, offsets = _compute_distances_offsets_numpy(cube_shape)

        # Precompute sorted indices by distance (excluding center)
        # Flatten, sort by distance, convert back to 3D indices
        flat_distances = distances.ravel()
        # Create all indices
        all_indices = np.arange(len(flat_distances))
        # Sort by distance (ascending), excluding center
        center_flat = center_t * dh_size * dw_size + center_h * dw_size + center_w
        mask = all_indices != center_flat
        sorted_flat = all_indices[mask][np.argsort(flat_distances[mask])]
        # Convert flat indices to 3D
        sorted_t = sorted_flat // (dh_size * dw_size)
        sorted_h = (sorted_flat % (dh_size * dw_size)) // dw_size
        sorted_w = sorted_flat % dw_size
        sorted_indices = np.stack([sorted_t, sorted_h, sorted_w], axis=-1).astype(
            np.int32
        )

        return {
            "quadrants": quads,
            "distances": distances,
            "offsets": offsets,
            "sorted_indices": sorted_indices,
        }

    def clear_cache(self) -> None:
        """Clear all cached templates."""
        self._cache.clear()

    @staticmethod
    def cube_size_for_ext(ext_range: int) -> int:
        """Compute the cube side length for a given extension range.

        The sub-cube spans [center - 6 - ext, center + 7 + ext) on each axis,
        so the size is 13 + 2 * ext.

        Args:
            ext_range: The EXT_RANGE value (6 in reference).

        Returns:
            Cube side length (same for all three dimensions).
        """
        return 13 + 2 * ext_range
