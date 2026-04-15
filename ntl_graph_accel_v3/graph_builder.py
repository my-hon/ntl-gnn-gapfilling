"""
Core graph construction for ntl_graph_accel_v3.

Implements the reference algorithm exactly:
  1. Sub-cube extraction with EXT_RANGE=6
  2. Quadrant assignment (Plan A, 6 quadrants)
  3. Round-robin node selection by Euclidean distance
  4. Three edge types: Bresenham occlusion, same-time/space auxiliary, self-loop
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ntl_graph_accel_v3.config import Config
from ntl_graph_accel_v3.template import TemplateManager


# ---------------------------------------------------------------------------
# Bresenham 3D line algorithm (matches reference exactly)
# ---------------------------------------------------------------------------

def bresenham_3d(
    start: Tuple[int, int, int], end: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """Compute 3D Bresenham line from start to end (inclusive).

    This is a straightforward 3D extension of the 2D Bresenham algorithm,
    matching the reference implementation exactly.

    Args:
        start: (t, h, w) start coordinates.
        end: (t, h, w) end coordinates.

    Returns:
        List of (t, h, w) points along the line, including start and end.
    """
    points = []
    t0, h0, w0 = start
    t1, h1, w1 = end

    dt = abs(t1 - t0)
    dh = abs(h1 - h0)
    dw = abs(w1 - w0)

    t_step = 1 if t0 < t1 else -1
    h_step = 1 if h0 < h1 else -1
    w_step = 1 if w0 < w1 else -1

    if dt >= dh and dt >= dw:
        # t is the driving axis
        err_h = 2 * dh - dt
        err_w = 2 * dw - dt
        t, h, w = t0, h0, w0
        for _ in range(dt + 1):
            points.append((t, h, w))
            if err_h > 0:
                h += h_step
                err_h -= 2 * dt
            if err_w > 0:
                w += w_step
                err_w -= 2 * dt
            err_h += 2 * dh
            err_w += 2 * dw
            t += t_step
    elif dh >= dt and dh >= dw:
        # h is the driving axis
        err_t = 2 * dt - dh
        err_w = 2 * dw - dh
        t, h, w = t0, h0, w0
        for _ in range(dh + 1):
            points.append((t, h, w))
            if err_t > 0:
                t += t_step
                err_t -= 2 * dh
            if err_w > 0:
                w += w_step
                err_w -= 2 * dh
            err_t += 2 * dt
            err_w += 2 * dw
            h += h_step
    else:
        # w is the driving axis
        err_t = 2 * dt - dw
        err_h = 2 * dh - dw
        t, h, w = t0, h0, w0
        for _ in range(dw + 1):
            points.append((t, h, w))
            if err_t > 0:
                t += t_step
                err_t -= 2 * dw
            if err_h > 0:
                h += h_step
                err_h -= 2 * dw
            err_t += 2 * dt
            err_h += 2 * dh
            w += w_step

    return points


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_blocks_through_line(
    data_array: np.ndarray,
    start: Tuple[int, int, int],
    center: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """Get valid (non-NaN) through points along the Bresenham line.

    Computes the Bresenham line from start to center, removes the start and
    end points, then filters out NaN positions.

    Args:
        data_array: The sub-cube data array (3D, float32 with NaN for invalid).
        start: (t, h, w) start position in the sub-cube.
        center: (t, h, w) center position in the sub-cube.

    Returns:
        List of valid (non-NaN) through points between start and center.
    """
    line = bresenham_3d(start, center)
    # Remove start and end points
    if len(line) <= 2:
        return []
    through_points = line[1:-1]

    # Filter out NaN positions
    valid_points = []
    for pt in through_points:
        t, h, w = pt
        # Check bounds
        if 0 <= t < data_array.shape[0] and \
           0 <= h < data_array.shape[1] and \
           0 <= w < data_array.shape[2]:
            val = data_array[t, h, w]
            if not np.isnan(val):
                valid_points.append(pt)

    return valid_points


def filter_triplets(
    node_pos_list: List[Tuple[int, int, int]],
    new_pos: Tuple[int, int, int],
    center: Tuple[int, int, int],
) -> List[int]:
    """Find existing nodes that share time or space with the new node.

    For each existing node in node_pos_list, check if it shares:
      - Same time (t) but different space (h, w), OR
      - Same space (h or w) but different time (t)

    Only include the existing node if the NEW node is CLOSER to center than
    the existing node.

    Args:
        node_pos_list: List of existing node positions (tuples of (t, h, w)).
        new_pos: Position of the newly added node.
        center: Center position (t, h, w).

    Returns:
        List of indices (into node_pos_list) of qualifying existing nodes.
    """
    new_t, new_h, new_w = new_pos
    ct, ch, cw = center

    # Distance of new node to center
    new_dist = np.sqrt(
        (new_t - ct) ** 2 + (new_h - ch) ** 2 + (new_w - cw) ** 2
    )

    result = []
    for idx, pos in enumerate(node_pos_list):
        exist_t, exist_h, exist_w = pos

        # Skip center node (index 0)
        if exist_t == ct and exist_h == ch and exist_w == cw:
            continue

        # Check same-time or same-space condition
        same_time_diff_space = (exist_t == new_t) and (
            exist_h != new_h or exist_w != new_w
        )
        same_space_diff_time = (
            (exist_h == new_h or exist_w == new_w) and exist_t != new_t
        )

        if same_time_diff_space or same_space_diff_time:
            # Check if new node is closer to center
            exist_dist = np.sqrt(
                (exist_t - ct) ** 2 + (exist_h - ch) ** 2 + (exist_w - cw) ** 2
            )
            if new_dist < exist_dist:
                result.append(idx)

    return result


# ---------------------------------------------------------------------------
# Core graph generation (matches reference algorithm exactly)
# ---------------------------------------------------------------------------

def graph_generate(
    data_array: np.ndarray,
    search_node: int,
    template_distances: np.ndarray,
    template_offsets: np.ndarray,
    template_quads: np.ndarray,
    sorted_indices: np.ndarray,
    edge_scale: float = 8.0,
) -> Dict[str, np.ndarray]:
    """Generate a single graph from a sub-cube data array.

    This is the core graph construction function that matches the reference
    algorithm exactly.

    Algorithm:
      1. Center node at index 0 with feature [-1.0]
      2. Sort all valid positions by Euclidean distance to center
      3. Round-robin quadrant selection (quad_cls_flag cycles 1-6)
      4. For each selected node, immediately build edges:
         - Type A: Bresenham occlusion edge
         - Type B: Same-time/space auxiliary edges
         - Type C: Self-loop
      5. Stop when we have search_node+1 total nodes

    Args:
        data_array: 3D float32 array (sub-cube), NaN for invalid pixels.
        search_node: Number of neighbor nodes to select.
        template_distances: Precomputed distance matrix.
        template_offsets: Precomputed offset matrix (..., 3).
        template_quads: Precomputed quadrant labels.
        sorted_indices: Precomputed sorted position indices (N, 3).
        edge_scale: Normalization factor for edge attributes.

    Returns:
        Dictionary with keys:
            - 'node_features': float32 array (N+1, 1)
            - 'edge_index': uint16 array (2, E)
            - 'edge_attr': float32 array (E, 3)
            - 'ground_truth': float32 array (N+1, 1)
    """
    cube_shape = data_array.shape
    center_t = cube_shape[0] // 2
    center_h = cube_shape[1] // 2
    center_w = cube_shape[2] // 2
    center = (center_t, center_h, center_w)

    # Initialize node lists
    node_features = [[-1.0]]  # Center node: feature = [-1.0] (prediction target)
    node_pos_list = [center]  # Position 0 is always center
    ground_truth = [[data_array[center_t, center_h, center_w]]]  # May be NaN

    # Edge lists
    edge_src = []
    edge_dst = []
    edge_attr_list = []

    # Build a set of existing positions for fast lookup
    node_pos_set = {center}

    # Round-robin quadrant flag: cycles 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 -> ...
    quad_cls_flag = 1

    # Total nodes needed
    total_nodes = search_node + 1  # center + search_node neighbors

    # Iterate through sorted positions by distance
    # sorted_indices is already sorted by distance, excluding center
    remaining = list(range(len(sorted_indices)))

    node_idx = 1  # Next node index to assign

    while remaining and node_idx < total_nodes:
        # We need to iterate through remaining positions to find one
        # whose quadrant matches quad_cls_flag
        found = False
        new_remaining = []

        for ri in remaining:
            if found:
                new_remaining.append(ri)
                continue

            pos_flat = sorted_indices[ri]
            pt = (int(pos_flat[0]), int(pos_flat[1]), int(pos_flat[2]))

            # Skip NaN positions
            val = data_array[pt[0], pt[1], pt[2]]
            if np.isnan(val):
                new_remaining.append(ri)
                continue

            # Check quadrant
            quad = int(template_quads[pt[0], pt[1], pt[2]])
            if quad == quad_cls_flag:
                # Select this node
                node_features.append([val])
                node_pos_list.append(pt)
                ground_truth.append([val])
                node_pos_set.add(pt)

                # ---- Build edges for this node IMMEDIATELY ----

                # Type A: Bresenham occlusion edge
                through_points = get_blocks_through_line(data_array, pt, center)
                if len(through_points) == 0:
                    # No valid through points: connect directly to center
                    edge_src.append(node_idx)
                    edge_dst.append(0)
                    # Edge attribute: offset from this node to center
                    offset = template_offsets[pt[0], pt[1], pt[2]]
                    edge_attr_list.append([
                        float(offset[0]),
                        float(offset[1]),
                        float(offset[2]),
                    ])
                else:
                    # Find the FIRST through point that is already in node_pos_list
                    connected = False
                    for tp in through_points:
                        if tp in node_pos_set:
                            # Connect to this existing node
                            target_idx = node_pos_list.index(tp)
                            edge_src.append(node_idx)
                            edge_dst.append(target_idx)
                            # Edge attribute: offset between them
                            dt = pt[0] - tp[0]
                            dh = pt[1] - tp[1]
                            dw = pt[2] - tp[2]
                            edge_attr_list.append([float(dt), float(dh), float(dw)])
                            connected = True
                            break
                    if not connected:
                        # Fallback: connect to center
                        edge_src.append(node_idx)
                        edge_dst.append(0)
                        offset = template_offsets[pt[0], pt[1], pt[2]]
                        edge_attr_list.append([
                            float(offset[0]),
                            float(offset[1]),
                            float(offset[2]),
                        ])

                # Type B: Same-time/space auxiliary edges
                aux_targets = filter_triplets(node_pos_list, pt, center)
                existing_edges = set()
                for s, d in zip(edge_src, edge_dst):
                    existing_edges.add((s, d))
                    existing_edges.add((d, s))

                for target_idx in aux_targets:
                    if target_idx == node_idx:
                        continue
                    edge_key = (node_idx, target_idx)
                    edge_key_rev = (target_idx, node_idx)
                    if edge_key not in existing_edges and \
                       edge_key_rev not in existing_edges:
                        edge_src.append(node_idx)
                        edge_dst.append(target_idx)
                        target_pos = node_pos_list[target_idx]
                        dt = pt[0] - target_pos[0]
                        dh = pt[1] - target_pos[1]
                        dw = pt[2] - target_pos[2]
                        edge_attr_list.append([float(dt), float(dh), float(dw)])
                        existing_edges.add(edge_key)
                        existing_edges.add(edge_key_rev)

                # Type C: Self-loop
                edge_src.append(node_idx)
                edge_dst.append(node_idx)
                edge_attr_list.append([0.0, 0.0, 0.0])

                # Advance to next node
                node_idx += 1
                found = True

                # Increment quad_cls_flag (wrap at 7 -> 1)
                quad_cls_flag += 1
                if quad_cls_flag > 6:
                    quad_cls_flag = 1
            else:
                new_remaining.append(ri)

        remaining = new_remaining

        # Safety: if we went through all remaining and found nothing,
        # break to avoid infinite loop
        if not found:
            break

    # Convert to numpy arrays
    node_features_arr = np.array(node_features, dtype=np.float32)
    ground_truth_arr = np.array(ground_truth, dtype=np.float32)

    if len(edge_src) > 0:
        edge_index_arr = np.array(
            [edge_src, edge_dst], dtype=np.uint16
        )
        edge_attr_arr = np.array(edge_attr_list, dtype=np.float32) / edge_scale
    else:
        # Fallback: create a minimal graph with just center
        edge_index_arr = np.zeros((2, 0), dtype=np.uint16)
        edge_attr_arr = np.zeros((0, 3), dtype=np.float32)

    return {
        "node_features": node_features_arr,
        "edge_index": edge_index_arr,
        "edge_attr": edge_attr_arr,
        "ground_truth": ground_truth_arr,
    }


# ---------------------------------------------------------------------------
# Process a single index position
# ---------------------------------------------------------------------------

def process_index(
    data_array: np.ndarray,
    pos: Tuple[int, int, int],
    search_node: int,
    templates: Dict[str, np.ndarray],
    ext_range: int = 6,
    edge_scale: float = 8.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Process a single center position and generate a graph.

    Extracts a sub-cube centered at the given position, then calls
    graph_generate to build the graph.

    Args:
        data_array: Full 3D data array (T, H, W).
        pos: Center position (tc, hc, wc).
        search_node: Number of neighbor nodes.
        templates: Precomputed template dict from TemplateManager.
        ext_range: Extension range (EXT_RANGE=6).
        edge_scale: Edge attribute normalization factor.

    Returns:
        Graph dict with node_features, edge_index, edge_attr, ground_truth,
        and position. Returns None if the sub-cube cannot be extracted.
    """
    tc, hc, wc = pos
    T, H, W = data_array.shape

    # Initial extension
    ext = ext_range

    # Cube size: 13 + 2*ext on each axis
    cube_size = 13 + 2 * ext

    # Check if initial extraction is valid
    t_start = tc - 6 - ext
    t_end = tc + 7 + ext
    h_start = hc - 6 - ext
    h_end = hc + 7 + ext
    w_start = wc - 6 - ext
    w_end = wc + 7 + ext

    # If out of bounds, increase ext until we can extract a valid cube
    # (or until we hit the data boundary)
    max_ext = max(
        -t_start if t_start < 0 else 0,
        t_end - T if t_end > T else 0,
        -h_start if h_start < 0 else 0,
        h_end - H if h_end > H else 0,
        -w_start if w_start < 0 else 0,
        w_end - W if w_end > W else 0,
    )

    if max_ext > 0:
        ext += max_ext
        cube_size = 13 + 2 * ext
        t_start = tc - 6 - ext
        t_end = tc + 7 + ext
        h_start = hc - 6 - ext
        h_end = hc + 7 + ext
        w_start = wc - 6 - ext
        w_end = wc + 7 + ext

    # Final bounds check
    if t_start < 0 or t_end > T or h_start < 0 or h_end > H or \
       w_start < 0 or w_end > W:
        return None

    # Extract sub-cube
    cube = data_array[t_start:t_end, h_start:h_end, w_start:w_end].copy()

    # Check if cube shape matches template
    expected_shape = (cube_size, cube_size, cube_size)
    if cube.shape != expected_shape:
        # Recompute templates for this cube size
        tm = TemplateManager()
        templates = tm.get_templates(cube.shape)

    # Generate graph
    graph = graph_generate(
        cube,
        search_node,
        templates["distances"],
        templates["offsets"],
        templates["quadrants"],
        templates["sorted_indices"],
        edge_scale,
    )

    # Add position
    graph["position"] = np.array([[tc, hc, wc]], dtype=np.uint16)

    return graph


# ---------------------------------------------------------------------------
# GraphBuilder class
# ---------------------------------------------------------------------------

class GraphBuilder:
    """High-level graph builder with batch processing support.

    Args:
        config: Configuration object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.template_manager = TemplateManager()

        # Precompute templates for the default cube size
        default_cube_size = 13 + 2 * config.ext_range
        default_shape = (default_cube_size, default_cube_size, default_cube_size)
        self._default_templates = self.template_manager.get_templates(default_shape)

    def build_single(
        self,
        data_array: np.ndarray,
        tc: int,
        hc: int,
        wc: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Build a single graph for the given center position.

        Args:
            data_array: Full 3D data array (T, H, W), float32 with NaN.
            tc: Time index of center pixel.
            hc: Height index of center pixel.
            wc: Width index of center pixel.

        Returns:
            Graph dict or None if extraction fails.
        """
        return process_index(
            data_array=data_array,
            pos=(tc, hc, wc),
            search_node=self.config.search_node,
            templates=self._default_templates,
            ext_range=self.config.ext_range,
            edge_scale=self.config.edge_scale,
        )

    def build_batch(
        self,
        data_array: np.ndarray,
        positions: List[Tuple[int, int, int]],
    ) -> List[Optional[Dict[str, np.ndarray]]]:
        """Build graphs for a batch of positions (sequential).

        For parallel processing, use joblib externally.

        Args:
            data_array: Full 3D data array (T, H, W), float32 with NaN.
            positions: List of (tc, hc, wc) tuples.

        Returns:
            List of graph dicts (None for failed extractions).
        """
        results = []
        for tc, hc, wc in positions:
            graph = self.build_single(data_array, tc, hc, wc)
            results.append(graph)
        return results
