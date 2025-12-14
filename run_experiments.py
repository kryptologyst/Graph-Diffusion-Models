#!/usr/bin/env python3
"""Script to run multiple experiments and create a model leaderboard."""

import subprocess
import json
import pandas as pd
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import setup_logging


def run_experiment(config_path: str, experiment_name: str) -> dict:
    """Run a single experiment.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger("experiments")
    logger.info(f"Running experiment: {experiment_name}")
    
    try:
        # Run training
        cmd = [
            "python", "train.py",
            "--config", config_path,
            "--experiment-name", experiment_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Experiment {experiment_name} failed: {result.stderr}")
            return {"error": result.stderr}
        
        # Load results
        results_path = Path("results") / f"{experiment_name}_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"No results file found for {experiment_name}")
            return {"error": "No results file"}
            
    except Exception as e:
        logger.error(f"Exception in experiment {experiment_name}: {e}")
        return {"error": str(e)}


def main():
    """Run multiple experiments and create leaderboard."""
    logger = setup_logging("INFO")
    
    # Define experiments
    experiments = [
        ("configs/default.yaml", "gcn_baseline"),
        ("configs/gat.yaml", "gat_baseline"),
        ("configs/default.yaml", "gcn_no_diffusion"),
        ("configs/gat.yaml", "gat_no_diffusion"),
    ]
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    for config_path, experiment_name in experiments:
        if config_path == "configs/default.yaml" and "no_diffusion" in experiment_name:
            # Modify command for no diffusion
            cmd = [
                "python", "train.py",
                "--config", config_path,
                "--experiment-name", experiment_name,
                "--no-diffusion"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Experiment {experiment_name} failed: {result.stderr}")
                all_results[experiment_name] = {"error": result.stderr}
                continue
            
            # Load results
            results_path = Path("results") / f"{experiment_name}_results.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    all_results[experiment_name] = json.load(f)
            else:
                all_results[experiment_name] = {"error": "No results file"}
        else:
            all_results[experiment_name] = run_experiment(config_path, experiment_name)
    
    # Create leaderboard
    leaderboard_data = []
    
    for experiment_name, results in all_results.items():
        if "error" in results:
            continue
            
        test_results = results.get("eval_results", {}).get("test", {})
        
        leaderboard_data.append({
            "Model": experiment_name,
            "Test Accuracy": test_results.get("accuracy", 0.0),
            "Test F1 (Macro)": test_results.get("f1_macro", 0.0),
            "Test F1 (Micro)": test_results.get("f1_micro", 0.0),
            "Test AUROC": test_results.get("auroc_macro", 0.0),
            "Test Loss": test_results.get("loss", float('inf')),
            "Parameters": results.get("train_results", {}).get("num_parameters", 0)
        })
    
    # Create DataFrame and save
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values("Test Accuracy", ascending=False)
        
        leaderboard_path = Path("results") / "model_leaderboard.csv"
        df.to_csv(leaderboard_path, index=False)
        
        logger.info("Model leaderboard created:")
        print(df.to_string(index=False))
        
        # Save detailed results
        detailed_path = Path("results") / "all_experiments.json"
        with open(detailed_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {detailed_path}")
    else:
        logger.error("No successful experiments to create leaderboard")


if __name__ == "__main__":
    main()
