import sys
import os
import argparse
import json
from config import SimulationConfig
from simulation import SimulationEngine


def main():
    # Set up command-line argument parser expecting a config JSON file path
    parser = argparse.ArgumentParser(description="Run gVXR simulation task in dedicated process")
    parser.add_argument("--config", required=True, help="Path to SimulationConfig JSON")
    args = parser.parse_args()

    # Validate that the configuration file exists 
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # JSON file to SimulationConfig object
    config = SimulationConfig.from_json(args.config)

    # Callback to stream real-time progress updates
    def progress_callback(curr: int, total: int, msg: str):
        print(f"PROGRESS: {curr}/{total}: {msg}", flush=True)

    # Execute simulation and handle success or failure outcomes
    try:
        result = SimulationEngine.run_simulation(config, progress_callback=progress_callback)
        # Notify the parent process of successful completion and output location
        print("SIMULATION_SUCCESS", flush=True)
        print(f"RESULT_DIR: {result['output_dir']}", flush=True)
    except Exception as e:
        # Terminate with an error code
        print(f"SIMULATION_FAILED: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
