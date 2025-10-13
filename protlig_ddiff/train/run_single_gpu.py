#!/usr/bin/env python3
"""
Simple wrapper script for single GPU training.
This script makes it easy to run training on a single GPU without DDP complexity.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Single GPU Training Wrapper")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--datafile", type=str, required=True, help="Path to training data")
    
    # Optional arguments with good defaults
    parser.add_argument("--work_dir", type=str, default="./single_gpu_run", help="Working directory")
    parser.add_argument("--device", type=str, default="0", help="GPU device (0, 1, 2, etc.) or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="single-gpu-training", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, help="Wandb run name (auto-generated if not specified)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_checkpoint", type=str, help="Resume from checkpoint")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.datafile):
        print(f"❌ Data file not found: {args.datafile}")
        return 1
    
    # Setup device string
    if args.device.lower() == 'cpu':
        device_str = 'cpu'
    elif args.device.isdigit():
        device_str = args.device
    else:
        device_str = args.device
    
    # Generate wandb name if not provided
    if not args.wandb_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_name = f"single-gpu-{timestamp}"
    
    # Print configuration
    print("🚀 Single GPU Training Configuration")
    print("=" * 50)
    print(f"Config file:    {args.config}")
    print(f"Data file:      {args.datafile}")
    print(f"Work directory: {args.work_dir}")
    print(f"Device:         {device_str}")
    print(f"Seed:           {args.seed}")
    print(f"Wandb project:  {args.wandb_project}")
    print(f"Wandb name:     {args.wandb_name}")
    print(f"Wandb enabled:  {not args.no_wandb}")
    if args.resume_checkpoint:
        print(f"Resume from:    {args.resume_checkpoint}")
    print()
    
    # Build command
    script_path = Path(__file__).parent / "protlig_ddiff" / "train" / "run_train_clean.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--config", args.config,
        "--datafile", args.datafile,
        "--work_dir", args.work_dir,
        "--device", device_str,
        "--seed", str(args.seed),
        "--wandb_project", args.wandb_project,
        "--wandb_name", args.wandb_name,
    ]
    
    if args.no_wandb:
        cmd.append("--no_wandb")
    
    if args.resume_checkpoint:
        cmd.extend(["--resume_checkpoint", args.resume_checkpoint])
    
    # Note: NO --cluster argument = single GPU mode
    
    print("🔧 Running command:")
    print(" ".join(cmd))
    print()
    
    # Set environment for single GPU training
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device_str if device_str != 'cpu' else ''
    
    # Run training
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
