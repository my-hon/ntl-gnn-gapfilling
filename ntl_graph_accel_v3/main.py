"""
CLI entry point for ntl_graph_accel_v3.

Modes:
  train  - Natural breaks sampling -> parallel graph generation -> save pkl
  predict - Day-by-day NaN processing -> parallel graph generation -> save pkl

Usage:
  python -m ntl_graph_accel_v3.main train --input data.tif --quality qual.tif --output ./output
  python -m ntl_graph_accel_v3.main predict --input data.tif --output ./output
"""

import argparse
import logging
import os
import pickle
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from ntl_graph_accel_v3.config import Config
from ntl_graph_accel_v3.data_loader import DataLoader
from ntl_graph_accel_v3.graph_builder import GraphBuilder

logger = logging.getLogger("ntl_graph_accel_v3")


# ---------------------------------------------------------------------------
# Parallel graph generation helpers
# ---------------------------------------------------------------------------

def _build_graph_for_position(
    args_tuple: Tuple,
) -> Optional[dict]:
    """Worker function for parallel graph generation.

    Args:
        args_tuple: (data_array, tc, hc, wc, search_node, templates,
                     ext_range, edge_scale)

    Returns:
        Graph dict or None.
    """
    from ntl_graph_accel_v3.graph_builder import process_index

    data_array, tc, hc, wc, search_node, templates, ext_range, edge_scale = args_tuple
    return process_index(
        data_array=data_array,
        pos=(tc, hc, wc),
        search_node=search_node,
        templates=templates,
        ext_range=ext_range,
        edge_scale=edge_scale,
    )


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

def run_train(config: Config) -> None:
    """Run training mode: natural breaks sampling + parallel graph generation.

    Steps:
      1. Load and preprocess data
      2. Sample positions using natural breaks
      3. Generate graphs in parallel using joblib
      4. Save results as pkl files

    Args:
        config: Configuration object.
    """
    logger.info("=== Training Mode ===")
    logger.info("Loading and preprocessing data...")

    loader = DataLoader(config)
    data = loader.get_preprocessed()
    logger.info(
        f"Data shape: {data.shape}, "
        f"NaN count: {np.isnan(data).sum()} / {data.size}"
    )

    # Save preprocessed data for potential reuse
    loader.save_preprocessed()
    logger.info(f"Preprocessed data saved to cache: {config.cache_dir}")

    # Sample positions
    logger.info("Sampling positions via natural breaks...")
    positions = loader.get_training_positions()
    logger.info(f"Total sampled positions: {len(positions)}")

    if len(positions) == 0:
        logger.warning("No valid positions found. Exiting.")
        return

    # Build graphs
    logger.info("Building graphs...")
    os.makedirs(config.output_dir, exist_ok=True)

    _run_parallel_graph_generation(
        data=data,
        positions=positions,
        config=config,
        output_dir=config.output_dir,
        prefix="train",
    )

    logger.info("Training mode complete.")


# ---------------------------------------------------------------------------
# Prediction mode
# ---------------------------------------------------------------------------

def run_predict(config: Config) -> None:
    """Run prediction mode: day-by-day NaN processing + parallel graph generation.

    Steps:
      1. Load and preprocess data
      2. Extract NaN positions per day
      3. For each day, generate graphs in parallel
      4. Save per-day pkl files

    Args:
        config: Configuration object.
    """
    logger.info("=== Prediction Mode ===")
    logger.info("Loading and preprocessing data...")

    loader = DataLoader(config)
    data = loader.get_preprocessed()
    logger.info(
        f"Data shape: {data.shape}, "
        f"NaN count: {np.isnan(data).sum()} / {data.size}"
    )

    # Extract NaN positions by day
    logger.info("Extracting NaN positions by day...")
    day_positions = loader.get_prediction_positions()
    logger.info(f"Days with NaN positions: {len(day_positions)}")

    if len(day_positions) == 0:
        logger.warning("No NaN positions found. Exiting.")
        return

    os.makedirs(config.output_dir, exist_ok=True)

    # Process each day
    for day_idx, (day, positions_2d) in enumerate(day_positions):
        logger.info(
            f"Processing day {day}: {len(positions_2d)} NaN positions "
            f"({day_idx + 1}/{len(day_positions)})"
        )

        # Convert (h, w) to (t, h, w)
        positions_3d = [(day, h, w) for h, w in positions_2d]

        _run_parallel_graph_generation(
            data=data,
            positions=positions_3d,
            config=config,
            output_dir=config.output_dir,
            prefix=f"predict_day_{day:04d}",
        )

    logger.info("Prediction mode complete.")


# ---------------------------------------------------------------------------
# Parallel graph generation
# ---------------------------------------------------------------------------

def _run_parallel_graph_generation(
    data: np.ndarray,
    positions: List[Tuple[int, int, int]],
    config: Config,
    output_dir: str,
    prefix: str,
) -> None:
    """Generate graphs in parallel and save results.

    Uses joblib.Parallel for multiprocessing, matching the reference approach.

    Args:
        data: Preprocessed float32 data array.
        positions: List of (t, h, w) center positions.
        config: Configuration object.
        output_dir: Directory to save output pkl files.
        prefix: Filename prefix for output files.
    """
    from ntl_graph_accel_v3.template import TemplateManager

    # Prepare templates
    tm = TemplateManager()
    default_cube_size = 13 + 2 * config.ext_range
    default_shape = (default_cube_size, default_cube_size, default_cube_size)
    templates = tm.get_templates(default_shape)

    # Prepare arguments for parallel execution
    # We pass the data array and config values; each worker gets a tuple
    # Note: data_array is shared (read-only), so this is memory-efficient
    args_list = [
        (data, tc, hc, wc, config.search_node, templates,
         config.ext_range, config.edge_scale)
        for tc, hc, wc in positions
    ]

    # Use joblib for parallel processing
    try:
        from joblib import Parallel, delayed

        logger.info(f"Starting parallel graph generation with {config.n_jobs} workers...")
        start_time = time.time()

        graphs = Parallel(n_jobs=config.n_jobs, verbose=10)(
            delayed(_build_graph_for_position)(args) for args in args_list
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Graph generation complete: {len(graphs)} graphs in {elapsed:.1f}s "
            f"({len(graphs) / max(elapsed, 0.001):.1f} graphs/s)"
        )

    except ImportError:
        logger.warning(
            "joblib not available, falling back to sequential processing"
        )
        start_time = time.time()

        graphs = []
        for i, args in enumerate(args_list):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Processing {i + 1}/{len(args_list)}...")
            graph = _build_graph_for_position(args)
            graphs.append(graph)

        elapsed = time.time() - start_time
        logger.info(
            f"Sequential graph generation complete: {len(graphs)} graphs "
            f"in {elapsed:.1f}s"
        )

    # Filter out None results
    valid_graphs = [g for g in graphs if g is not None]
    logger.info(
        f"Valid graphs: {len(valid_graphs)}/{len(graphs)} "
        f"({len(graphs) - len(valid_graphs)} failed)"
    )

    # Save results
    output_path = os.path.join(output_dir, f"{prefix}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(valid_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved {len(valid_graphs)} graphs to {output_path} ({file_size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="ntl_graph_accel_v3 - Spatiotemporal Graph Construction "
                    "for NTL Gap-Filling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    subparsers.required = True

    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--input", "-i", type=str, required=True,
            help="Path to input NTL TIF/NumPy file",
        )
        p.add_argument(
            "--quality", "-q", type=str, default="",
            help="Path to quality flag TIF/NumPy file",
        )
        p.add_argument(
            "--output", "-o", type=str, default="./output_v3",
            help="Output directory for pkl files",
        )
        p.add_argument(
            "--cache", type=str, default="./cache_v3",
            help="Cache directory for intermediate files",
        )
        p.add_argument(
            "--n-jobs", "-j", type=int, default=8,
            help="Number of parallel workers",
        )
        p.add_argument(
            "--search-node", "-n", type=int, default=32,
            help="Number of neighbor nodes per graph",
        )
        p.add_argument(
            "--ext-range", type=int, default=6,
            help="Extension range for sub-cube extraction (EXT_RANGE)",
        )
        p.add_argument(
            "--edge-scale", type=float, default=8.0,
            help="Edge attribute normalization factor",
        )
        p.add_argument(
            "--sample-per-class", type=int, default=20000,
            help="Number of samples per natural-breaks category (training)",
        )
        p.add_argument(
            "--seed", type=int, default=0,
            help="Random seed for reproducibility",
        )
        p.add_argument(
            "--data-shape", type=str, default="366,560,666",
            help="Data shape as T,H,W",
        )
        p.add_argument(
            "--edge-time", type=int, default=50,
            help="Temporal buffer size (EDGE_TIME)",
        )
        p.add_argument(
            "--edge-height", type=int, default=50,
            help="Height buffer size (EDGE_HEIGHT)",
        )
        p.add_argument(
            "--edge-width", type=int, default=50,
            help="Width buffer size (EDGE_WIDTH)",
        )
        p.add_argument(
            "--verbose", "-v", action="store_true",
            help="Enable verbose logging",
        )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train", help="Training mode: natural breaks sampling + graph generation"
    )
    add_common_args(train_parser)

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict", help="Prediction mode: NaN position graph generation"
    )
    add_common_args(predict_parser)

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config object from parsed CLI arguments.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Configuration object.
    """
    data_shape = tuple(int(x) for x in args.data_shape.split(","))

    return Config(
        data_shape=data_shape,
        edge_time=args.edge_time,
        edge_height=args.edge_height,
        edge_width=args.edge_width,
        ext_range=args.ext_range,
        search_node=args.search_node,
        edge_scale=args.edge_scale,
        sample_per_class=args.sample_per_class,
        n_jobs=args.n_jobs,
        seed=args.seed,
        input_path=args.input,
        quality_path=args.quality,
        output_dir=args.output,
        cache_dir=args.cache,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    config = build_config_from_args(args)

    logger.info(f"ntl_graph_accel_v3 v3.0.0")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {config}")

    # Run appropriate mode
    if args.mode == "train":
        run_train(config)
    elif args.mode == "predict":
        run_predict(config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
