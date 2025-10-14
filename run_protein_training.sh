#!/bin/bash

# Protein Discrete Diffusion Training Script
# This script provides convenient commands for different training scenarios

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths (modify these for your setup)
CONFIG_FILE="/lus/flare/projects/FoundEpidem/xlian/IDEAL/Diffusion/prot_lig_discdiff/config_protein.yaml"
DATA_FILE="/lus/flare/projects/FoundEpidem/xlian/IDEAL/Diffusion/input_data/processed_pubchem_subset_50k.pt"  # CHANGE THIS
WORK_DIR="./experiments/protein_sedd_$(date +%Y%m%d_%H%M%S)"

# Default settings
DEVICE="cpu"
CLUSTER=""
WANDB_PROJECT="protein-discrete-diffusion"
WANDB_NAME="disc-diff-$(date +%Y%m%d_%H%M%S)"
RESUME_CHECKPOINT=""
SEED=42

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data PATH          Path to training data file (required)"
    echo "  --config PATH        Path to config file (default: $CONFIG_FILE)"
    echo "  --work_dir PATH      Working directory (default: auto-generated)"
    echo "  --device DEVICE      Device to use (default: $DEVICE)"
    echo "  --cluster TYPE       Cluster type: aurora, polaris (default: none)"
    echo "  --wandb_project NAME Wandb project name (default: $WANDB_PROJECT)"
    echo "  --wandb_name NAME    Wandb run name (default: auto-generated)"
    echo "  --resume PATH        Resume from checkpoint"
    echo "  --no_wandb           Disable wandb logging"
    echo "  --seed NUM           Random seed (default: $SEED)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic CPU training"
    echo "  $0 --data /path/to/data.jsonl"
    echo ""
    echo "  # GPU training with custom config"
    echo "  $0 --data /path/to/data.jsonl --device cuda:0 --config my_config.yaml"
    echo ""
    echo "  # Aurora cluster training"
    echo "  $0 --data /path/to/data.jsonl --cluster aurora --device xpu:0"
    echo ""
    echo "  # Resume training"
    echo "  $0 --data /path/to/data.jsonl --resume ./experiments/run1/checkpoints/checkpoint_step_10000.pt"
    echo ""
}

check_requirements() {
    echo "🔍 Checking requirements..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python."
        exit 1
    fi
    
    # Check if config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        echo "💡 Create a config file or specify one with --config"
        exit 1
    fi
    
    # Check if data file exists
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "❌ Data file not found: $DATA_FILE"
        echo "💡 Specify the correct data file with --data"
        exit 1
    fi
    
    echo "✅ Requirements check passed"
}

print_config_summary() {
    echo ""
    echo "🧬 PROTEIN DISCRETE DIFFUSION TRAINING CONFIGURATION"
    echo "========================================"
    echo "📁 Data file:      $DATA_FILE"
    echo "⚙️  Config file:    $CONFIG_FILE"
    echo "📂 Work directory: $WORK_DIR"
    echo "🖥️  Device:         $DEVICE"
    echo "🌐 Cluster:        ${CLUSTER:-"none"}"
    echo "📊 Wandb project:  $WANDB_PROJECT"
    echo "🏷️  Wandb name:     $WANDB_NAME"
    echo "🎲 Seed:           $SEED"
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        echo "🔄 Resume from:    $RESUME_CHECKPOINT"
    fi
    echo "========================================"
    echo ""
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --no_wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    echo "🚀 Starting Protein Discrete Diffusion Training"
    echo "=================================="
    
    # Check requirements
    check_requirements
    
    # Print configuration
    print_config_summary
    
    # Create work directory
    mkdir -p "$WORK_DIR"
    echo "📁 Created work directory: $WORK_DIR"
    
    # Copy config to work directory for reproducibility
    cp "$CONFIG_FILE" "$WORK_DIR/config.yaml"
    echo "📋 Copied config to work directory"
    
    # Build command
    CMD="python /lus/flare/projects/FoundEpidem/xlian/IDEAL/Diffusion/prot_lig_discdiff/protlig_ddiff/train/run_train_clean.py"
    CMD="$CMD --config $CONFIG_FILE"
    CMD="$CMD --datafile $DATA_FILE"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --device $DEVICE"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --wandb_project $WANDB_PROJECT"
    CMD="$CMD --wandb_name $WANDB_NAME"
    
    # Add cluster if specified
    if [[ -n "$CLUSTER" ]]; then
        CMD="$CMD --cluster $CLUSTER"
    fi
    
    # Add resume if specified
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        CMD="$CMD --resume_checkpoint $RESUME_CHECKPOINT"
    fi
    
    # Add no_wandb if specified
    if [[ -n "$NO_WANDB" ]]; then
        CMD="$CMD $NO_WANDB"
    fi
    
    echo "🔧 Command: $CMD"
    echo ""
    
    # Save command to work directory
    echo "$CMD" > "$WORK_DIR/command.txt"
    
    # Run training
    echo "🚀 Starting training..."
    echo "📝 Logs will be saved to: $WORK_DIR"
    echo "⏰ Started at: $(date)"
    echo ""
    
    # Execute the command
    eval "$CMD" 2>&1 | tee "$WORK_DIR/training.log"
    
    # Check if training completed successfully
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo ""
        echo "🎉 Training completed successfully!"
        echo "📁 Results saved to: $WORK_DIR"
        echo "⏰ Completed at: $(date)"
    else
        echo ""
        echo "❌ Training failed!"
        echo "📝 Check the logs in: $WORK_DIR/training.log"
        exit 1
    fi
}

# Run main function
main "$@"
