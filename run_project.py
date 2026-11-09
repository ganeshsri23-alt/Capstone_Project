"""
Instamart Capstone Project Runner
Runs project scripts sequentially from the project root.
"""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def run_script(script_name):
    script_path = ROOT / "src" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_name}")
    print(f"{'=' * 60}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        check=True
    )
def main():
    print("INSTAMART CAPSTONE PROJECT")
    print("Project root:", ROOT)
    # Generate raw data first
    print("\n========== GENERATING RAW DATA ==========")
    subprocess.run(
        [sys.executable, str(ROOT / "generate_data.py")],
        cwd=ROOT,
        check=True
    )
    # PART A: DATA FOUNDATION
    print("\n========== PART A ==========")
    run_script("data_pipeline.py")
    run_script("store_validate.py")
    run_script("run_sql.py")
    run_script("eda_analysis.py")
    # PART B: CUSTOMER SEGMENTATION
    print("\n========== PART B ==========")
    run_script("customer_segmentation.py")
    # PART C: TICKET TRIAGE
    print("\n========== PART C ==========")
    run_script("ticket_triage.py")
    # MAIN PROJECT SCRIPT
    print("\n========== MAIN PROJECT ==========")
    run_script("instamart.py")
    print("\nPROJECT EXECUTION COMPLETED!")
if __name__ == "__main__":
    main()