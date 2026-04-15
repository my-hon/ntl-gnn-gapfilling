"""
Configuration for ntl_graph_accel_v3.

All hyperparameters and paths used in the graph construction pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Config:
    """Configuration for spatiotemporal graph construction.

    Attributes:
        data_shape: Shape of the input data array (time, height, width).
        edge_time: Buffer size for temporal dimension (EDGE_TIME in reference).
        edge_height: Buffer size for height dimension.
        edge_width: Buffer size for width dimension.
        ext_range: Extension range for sub-cube extraction (EXT_RANGE=6 in reference).
        search_node: Number of neighbor nodes to select per graph.
        edge_scale: Normalization factor for edge attributes (divide offsets by this).
        sample_per_class: Number of samples per natural-breaks category for training.
        natural_breaks: Bin boundaries for natural-breaks stratified sampling.
        n_jobs: Number of parallel workers for joblib.
        seed: Random seed for reproducibility.
        input_path: Path to the input NTL TIF/NumPy file.
        quality_path: Path to the quality flag TIF/NumPy file.
        output_dir: Directory for output pkl files.
        cache_dir: Directory for intermediate cache files.
    """

    # Data dimensions
    data_shape: Tuple[int, int, int] = (366, 560, 666)

    # Buffer sizes (EDGE_TIME, EDGE_HEIGHT, EDGE_WIDTH in reference)
    edge_time: int = 50
    edge_height: int = 50
    edge_width: int = 50

    # Sub-cube extraction
    ext_range: int = 6  # EXT_RANGE = 6 in reference (NOT 4 or 50!)

    # Graph construction
    search_node: int = 32
    edge_scale: float = 8.0

    # Training sampling
    sample_per_class: int = 20000
    natural_breaks: List[float] = field(default_factory=lambda: [
        -np.inf, 0.001, 0.00325, 0.0065, 0.0125, 0.025, 0.1, 0.25, 0.5, 1.0, np.inf
    ])

    # Parallelism
    n_jobs: int = 8
    seed: int = 0

    # Paths
    input_path: str = ""
    quality_path: str = ""
    output_dir: str = "./output_v3"
    cache_dir: str = "./cache_v3"

    @property
    def valid_slice(self) -> Tuple[slice, slice, slice]:
        """Return the valid region slice (excluding edge buffers)."""
        return (
            slice(self.edge_time, self.data_shape[0] - self.edge_time),
            slice(self.edge_height, self.data_shape[1] - self.edge_height),
            slice(self.edge_width, self.data_shape[2] - self.edge_width),
        )

    @property
    def num_categories(self) -> int:
        """Number of natural-breaks categories."""
        return len(self.natural_breaks) - 1
