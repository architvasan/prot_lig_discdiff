#!/bin/bash

# Fix for AF_UNIX path too long error on Polaris
# This script sets up a shorter temporary directory path

echo "🔧 Setting up temporary directory fix for Polaris..."

# Function to setup shorter temp directory
setup_temp_dir() {
    local base_dirs=("/tmp" "/dev/shm" ".")
    
    for base_dir in "${base_dirs[@]}"; do
        if [[ -d "$base_dir" && -w "$base_dir" ]]; then
            # Create unique temp directory
            local temp_dir=$(mktemp -d -p "$base_dir" pytorch_XXXXXX 2>/dev/null)
            if [[ $? -eq 0 && -d "$temp_dir" ]]; then
                export TMPDIR="$temp_dir"
                export TEMP="$temp_dir"
                export TMP="$temp_dir"
                echo "✅ Set TMPDIR to: $temp_dir"
                echo "📏 Path length: ${#temp_dir}"
                return 0
            fi
        fi
    done
    
    # Fallback to current directory
    local fallback_dir="./tmp_pytorch_$$"
    mkdir -p "$fallback_dir"
    export TMPDIR="$fallback_dir"
    export TEMP="$fallback_dir"
    export TMP="$fallback_dir"
    echo "⚠️  Using fallback temp dir: $fallback_dir"
    echo "📏 Path length: ${#fallback_dir}"
}

# Show current state
echo "Current TMPDIR: ${TMPDIR:-'Not set'}"
echo "Current path length: ${#TMPDIR}"

# Setup new temp directory
setup_temp_dir

# Additional multiprocessing settings for HPC systems
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Show final state
echo "New TMPDIR: $TMPDIR"
echo "New path length: ${#TMPDIR}"

# Function to cleanup temp directory
cleanup_temp_dir() {
    if [[ -n "$TMPDIR" && "$TMPDIR" != "/tmp" && -d "$TMPDIR" ]]; then
        echo "🧹 Cleaning up temporary directory: $TMPDIR"
        rm -rf "$TMPDIR"
    fi
}

# Set trap to cleanup on exit
trap cleanup_temp_dir EXIT

echo "🚀 Environment ready! You can now run your training script."
echo "   Example: python protlig_ddiff/train/run_train_clean.py --config config.yaml --datafile data.pt"
echo ""
echo "💡 To use this fix in your own script, source this file:"
echo "   source fix_polaris_tmpdir.sh"
