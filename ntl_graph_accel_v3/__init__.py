"""
ntl_graph_accel_v3 - Spatiotemporal Graph Construction for NTL Gap-Filling

Rewritten to match the reference implementation exactly, with Numba JIT acceleration
where applicable.

Graph construction algorithm:
  1. Data preprocessing: 65535->NaN, quality filter, /10.0, /100.0
  2. Sub-cube extraction with EXT_RANGE=6
  3. Quadrant assignment (Plan A, 6 quadrants)
  4. Round-robin node selection by distance
  5. Three types of edges: Bresenham occlusion, same-time/space auxiliary, self-loop
"""

__version__ = "3.0.0"

from ntl_graph_accel_v3.config import Config
from ntl_graph_accel_v3.graph_builder import GraphBuilder
from ntl_graph_accel_v3.template import TemplateManager
from ntl_graph_accel_v3.data_loader import DataLoader

__all__ = [
    "Config",
    "GraphBuilder",
    "TemplateManager",
    "DataLoader",
]
