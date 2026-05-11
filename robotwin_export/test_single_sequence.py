#!/usr/bin/env python3
"""
Test script to run the depth computation and RoboTwin export on a single DROID sequence.
This verifies that the entire pipeline works end-to-end on your server.
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test RoboTwin conversion on a single DROID sequence.")
    parser.add_argument("--scene_path", type=str, 
                        default="gs://gresearch/robotics/droid_raw/1.0.1/PennPAL/failure/2023-10-01/Sun_Oct__1_17:00:10_2023/",
                        help="GCS path to a single DROID scene.")
    args = parser.parse_args()

    # Get repository root
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(this_dir, ".."))
    
    # Define paths
    test_txt = os.path.join(repo_root, "robotwin_export", "test_path.txt")
    pw_out_dir = os.path.join(repo_root, "test_pointworld_out")
    rw_out_dir = os.path.join(repo_root, "test_robotwin_out")
    
    # 1. Write the single path to a temporary txt file
    print(f"--- Preparing test for scene: {args.scene_path} ---")
    with open(test_txt, "w") as f:
        f.write(args.scene_path + "\n")
        
    # Set PYTHONPATH so the 'real' module can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    # 2. Run compute_depth.py
    print("\n[1/2] Running depth computation (compute_depth.py)...")
    cmd_depth = [
        "python", os.path.join("real", "compute_depth.py"),
        "--input", test_txt,
        "--output_dir", pw_out_dir,
        "--allow_gcs_streaming"  # Streaming mode for test
    ]
    print(f"Command: {' '.join(cmd_depth)}")
    subprocess.run(cmd_depth, cwd=repo_root, env=env, check=True)
    
    # 3. Run convert_droid_to_robotwin.py
    print("\n[2/2] Running RoboTwin converter (convert_droid_to_robotwin.py)...")
    cmd_convert = [
        "python", os.path.join("robotwin_export", "convert_droid_to_robotwin.py"),
        "--input", test_txt,
        "--pointworld_output_dir", pw_out_dir,
        "--output_dir", rw_out_dir,
        "--allow_gcs_streaming"  # Streaming mode for test
    ]
    print(f"Command: {' '.join(cmd_convert)}")
    subprocess.run(cmd_convert, cwd=repo_root, env=env, check=True)
    
    print("\n--- Test Completed Successfully! ---")
    print(f"The PointWorld intermediate files (depth H5) are in: {pw_out_dir}")
    print(f"The final RoboTwin HDF5 file is in: {os.path.join(rw_out_dir, 'data')}")
    print("\nYou can now use read_hdf5.py to verify the generated HDF5 file:")
    print(f"  python path/to/read_hdf5.py {os.path.join(rw_out_dir, 'data', 'episode0.hdf5')}")

if __name__ == "__main__":
    main()
