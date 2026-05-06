"""
Multi-seed runner: train v3, v8 bypass, and twostage with multiple seeds.
Reports mean ± std of MAE across seeds.

Usage:
    python run_multiseed.py
    python run_multiseed.py --models v8 twostage   # run specific models only
"""

import argparse
import subprocess
import sys
import numpy as np

from config import SEEDS, DEFAULT_SEED, ensure_dirs
from utils import load_results

ensure_dirs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["v3", "v8", "twostage"],
                        choices=["v3", "v8", "twostage"])
    args = parser.parse_args()

    script_map = {
        "v3": "run_v3_baseline.py",
        "v8": "run_v8_bypass.py",
        "twostage": "run_twostage.py",
    }
    result_prefix_map = {
        "v3": "test_results_v3_seed{seed}.pkl",
        "v8": "test_results_v8_seed{seed}.pkl",
        "twostage": "test_results_twostage_seed{seed}.pkl",
    }

    # Run each model with each seed
    for model in args.models:
        script = script_map[model]
        for seed in SEEDS:
            print(f"\n{'#'*60}")
            print(f"  Running {model} with seed={seed}")
            print(f"{'#'*60}\n")
            cmd = [sys.executable, script, "--seed", str(seed)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  WARNING: {model} seed={seed} failed with code {result.returncode}")

    # Collect and summarise
    print(f"\n\n{'='*70}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Seeds':>6} {'Mean MAE':>12} {'Std MAE':>12} {'MAEs':>30}")
    print(f"{'-'*70}")

    for model in args.models:
        maes = []
        for seed in SEEDS:
            pkl_name = result_prefix_map[model].format(seed=seed)
            try:
                data = load_results(pkl_name)
                if model == "twostage":
                    mae = data["combined_test_mae"]
                else:
                    mae = data["mean_mae"]
                maes.append(mae)
            except FileNotFoundError:
                print(f"  Missing: {pkl_name}")

        if maes:
            maes_arr = np.array(maes)
            mae_str = ", ".join(f"{m:.4f}" for m in maes)
            print(f"  {model:<13} {len(maes):>6} {np.mean(maes_arr):>12.6f} "
                  f"{np.std(maes_arr):>12.6f} [{mae_str}]")

    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
