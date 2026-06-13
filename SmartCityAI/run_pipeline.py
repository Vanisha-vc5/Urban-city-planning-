"""
run_pipeline.py
===============
Phase 11 - Deployment: Complete Pipeline Runner

Master script that runs all SmartCityAI phases end-to-end.
Can be used for CI/CD, fresh environment setup, or demo execution.

USAGE:
    python run_pipeline.py                  # Run all phases
    python run_pipeline.py --phase 5 6 7   # Run specific phases
    python run_pipeline.py --demo           # Demo mode (fast, synthetic data)
    python run_pipeline.py --city "Delhi, India" --infra hospital school

PHASES:
    1: Download data (OSM, WorldPop, healthsites)
    2: Create H3 zones (create_zones.py)
    3: Feature engineering (build_feature_table.py)
    4: ML models (kmeans, svm, decision_tree, regression)
    5: Classical AI (best_first, hill_climbing, astar)
    6: Explanations (explanation_engine)
    7: Launch dashboard (streamlit app.py)
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Configuration ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

BASE_DIR  = Path(__file__).resolve().parent
SRC_DIR   = BASE_DIR / "src"


# ── Phase Definitions ──────────────────────────────────────────────────────────

PHASES = {
    1: {
        "name": "Data Download",
        "scripts": [
            SRC_DIR / "data_collection" / "download_osm.py",
            SRC_DIR / "data_collection" / "download_worldpop.py",
            SRC_DIR / "data_collection" / "download_healthsites.py",
            SRC_DIR / "data_collection" / "download_ev_data.py",
            SRC_DIR / "data_collection" / "download_census.py",
        ],
        "critical": False,  # Network-dependent, may fail
    },
    2: {
        "name": "Zone Creation (H3 Grid)",
        "scripts": [SRC_DIR / "preprocessing" / "create_zones.py"],
        "critical": True,
    },
    3: {
        "name": "Feature Engineering",
        "scripts": [SRC_DIR / "features" / "build_feature_table.py"],
        "critical": True,
    },
    4: {
        "name": "ML Models",
        "scripts": [
            SRC_DIR / "models" / "kmeans.py",
            SRC_DIR / "models" / "svm_priority.py",
            SRC_DIR / "models" / "decision_tree.py",
            SRC_DIR / "models" / "population_regression.py",
        ],
        "critical": True,
    },
    5: {
        "name": "Classical AI Optimization",
        "scripts": [
            SRC_DIR / "optimization" / "best_first.py",
            SRC_DIR / "optimization" / "hill_climbing.py",
            SRC_DIR / "optimization" / "astar_routing.py",
        ],
        "critical": True,
    },
    6: {
        "name": "Explanation Engine",
        "scripts": [SRC_DIR / "explainability" / "explanation_engine.py"],
        "critical": False,
    },
    7: {
        "name": "Dashboard Launch",
        "scripts": ["__dashboard__"],  # Special case: streamlit
        "critical": False,
    },
}

# ── Phase Runner ───────────────────────────────────────────────────────────────

def run_script(script_path: Path, extra_args: List[str] = None) -> bool:
    """
    Run a Python script as a subprocess.

    Args:
        script_path: Path to .py script
        extra_args:  Additional CLI arguments

    Returns:
        True if succeeded (exit code 0)
    """
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    log.info(f"  Running: {script_path.name} {' '.join(extra_args or [])}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        log.info(f"  ✅ {script_path.name} completed in {elapsed:.1f}s")
        return True
    else:
        log.error(f"  ❌ {script_path.name} failed (exit {result.returncode}) in {elapsed:.1f}s")
        return False


def run_phase(
    phase_num: int,
    extra_args: Optional[List[str]] = None,
    infra_types: Optional[List[str]] = None,
) -> bool:
    """
    Run all scripts in a phase.

    Args:
        phase_num:   Phase number (1-7)
        extra_args:  Additional args passed to each script
        infra_types: Infrastructure types for best_first.py

    Returns:
        True if all scripts succeeded
    """
    phase = PHASES.get(phase_num)
    if not phase:
        log.error(f"Unknown phase: {phase_num}")
        return False

    log.info(f"\n{'='*60}")
    log.info(f"Phase {phase_num}: {phase['name']}")
    log.info(f"{'='*60}")

    all_ok = True

    for script in phase["scripts"]:
        if script == "__dashboard__":
            # Special case: launch Streamlit
            dashboard_path = BASE_DIR / "dashboard" / "app.py"
            cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
            log.info(f"  Launching Streamlit dashboard: {dashboard_path}")
            log.info(f"  Command: {' '.join(cmd)}")
            log.info("  Visit http://localhost:8501 in your browser")
            subprocess.Popen(cmd)
            return True

        if not Path(script).exists():
            log.warning(f"  ⚠️  Script not found: {script} (skipping)")
            continue

        # Extra args per script
        script_args = list(extra_args or [])

        # best_first.py: run for each infra type
        if "best_first" in str(script) and infra_types:
            success = True
            for infra in infra_types:
                ok = run_script(script, script_args + ["--infra", infra, "--top", "20"])
                success = success and ok
            all_ok = all_ok and success
        else:
            ok = run_script(script, script_args)
            if not ok and phase["critical"]:
                log.error(f"  Critical phase {phase_num} script failed. Stopping.")
                return False
            all_ok = all_ok and ok

    return all_ok


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmartCityAI Complete Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                  # Run all phases
  python run_pipeline.py --phase 4 5     # Only ML + AI phases
  python run_pipeline.py --demo          # Fast demo mode
  python run_pipeline.py --city "Delhi, India"
        """,
    )
    parser.add_argument("--phase", nargs="+", type=int, default=None,
                        help="Specific phases to run (1-7)")
    parser.add_argument("--city",  default="Mumbai, India",
                        help="City name for data download")
    parser.add_argument("--infra", nargs="+", default=["hospital"],
                        choices=["hospital", "school", "ev_station", "fire_station", "all"],
                        help="Infrastructure types to rank")
    parser.add_argument("--demo",  action="store_true",
                        help="Demo mode: skip data download, use synthetic data")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip Streamlit dashboard launch")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("SmartCityAI — Full Pipeline Runner")
    log.info(f"City: {args.city}")
    log.info(f"Infrastructure: {args.infra}")
    log.info("=" * 70)

    # Determine phases to run
    if args.phase:
        phases_to_run = sorted(args.phase)
    elif args.demo:
        phases_to_run = [2, 3, 4, 5, 6, 7]  # Skip data download
    else:
        phases_to_run = list(range(1, 8))

    if args.skip_dashboard and 7 in phases_to_run:
        phases_to_run.remove(7)

    # Handle "all" infra type
    infra_types = (
        ["hospital", "school", "ev_station", "fire_station"]
        if "all" in args.infra else args.infra
    )

    pipeline_start = time.time()
    results = {}

    for phase_num in phases_to_run:
        # Skip phase 1 in demo mode
        if args.demo and phase_num == 1:
            log.info("⏭️  Skipping Phase 1 (demo mode — using synthetic data)")
            continue

        phase_args = []
        if phase_num == 1:
            phase_args = ["--city", args.city]

        ok = run_phase(phase_num, extra_args=phase_args, infra_types=infra_types)
        results[phase_num] = "✅ Success" if ok else "❌ Failed"

    # Summary
    elapsed = time.time() - pipeline_start
    log.info(f"\n{'='*60}")
    log.info("PIPELINE SUMMARY")
    log.info(f"{'='*60}")
    for phase_num, status in results.items():
        phase_name = PHASES.get(phase_num, {}).get("name", "Unknown")
        log.info(f"  Phase {phase_num}: {phase_name:30s} {status}")
    log.info(f"\nTotal time: {elapsed:.1f}s")

    if 7 not in phases_to_run:
        log.info("\n💡 Launch dashboard: streamlit run dashboard/app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
