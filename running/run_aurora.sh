#!/bin/bash

# Polaris Cluster Training Script for Protein Discrete Diffusion
# Optimized for NVIDIA GPUs with MPI support
#
# QUICK START:
# 1. Edit the HARDCODED SETTINGS section below with your paths
# 2. Run: ./run_polaris.sh --submit
#
# For help: ./run_polaris.sh --help

set -e

# =============================================================================
# POLARIS-SPECIFIC CONFIGURATION
# =============================================================================

# *** HARDCODED SETTINGS - MODIFY THESE FOR YOUR SETUP ***
# Set these to your actual paths and preferences to avoid specifying them every time
#
# EXAMPLE CONFIGURATION:
# PROJECT_ROOT="/eagle/YourProject/username/prot_lig_discdiff"
# HARDCODED_CONFIG_FILE="${PROJECT_ROOT}/config_protein.yaml"
# HARDCODED_DATA_FILE="${PROJECT_ROOT}/data/processed_uniref50.pt"
# HARDCODED_ACCOUNT="YourAllocation"

# Project paths (MODIFY THESE TO YOUR ACTUAL PATHS)
PROJECT_ROOT="/flare/FoundEpidem/avasan/IDEAL/Diffusion/prot_lig_discdiff"
HARDCODED_CONFIG_FILE="${PROJECT_ROOT}/configs/config_protein.yaml"
HARDCODED_DATA_FILE="${PROJECT_ROOT}/input_data/processed_uniref50.pt"
HARDCODED_WORK_DIR="${PROJECT_ROOT}/experiments/protligdiff_$(date +%Y%m%d_%H%M%S)"

# Job settings (MODIFY AS NEEDED)
HARDCODED_ACCOUNT="FoundEpidem"  # *** SET YOUR AURORA ACCOUNT/ALLOCATION HERE ***
HARDCODED_TIME_LIMIT=6  # Hours
HARDCODED_QUEUE="prod"

# Training settings (MODIFY AS NEEDED)
HARDCODED_WANDB_PROJECT="protein-discrete-diffusion-aurora"
HARDCODED_WANDB_NAME="aurora-run-$(date +%Y%m%d_%H%M%S)"
HARDCODED_NODES=20
HARDCODED_PPN=12  # Processes per node (Polaris has 4 GPUs per node)
HARDCODED_SEED=0310

# Advanced settings (usually don't need to change)
HARDCODED_DEVICE="xpu:0"
HARDCODED_CLUSTER="aurora"

# *** END HARDCODED SETTINGS ***

# Default values (will be overridden by hardcoded values if set, or command line args)
CONFIG_FILE="${HARDCODED_CONFIG_FILE:-config_protein.yaml}"
DATA_FILE="${HARDCODED_DATA_FILE:-}"
WORK_DIR="${HARDCODED_WORK_DIR:-./experiments/protligdiff_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${HARDCODED_WANDB_PROJECT:-protein-discrete-diffusion-aurora}"
WANDB_NAME="${HARDCODED_WANDB_NAME:-aurora-run-$(date +%Y%m%d_%H%M%S)}"
NODES="${HARDCODED_NODES:-1}"
PPN="${HARDCODED_PPN:-4}"
TIME_LIMIT="${HARDCODED_TIME_LIMIT:-2}"
QUEUE="${HARDCODED_QUEUE:-workq}"
ACCOUNT="${HARDCODED_ACCOUNT:-}"
DEVICE="${HARDCODED_DEVICE:-xpu:0}"
CLUSTER="${HARDCODED_CLUSTER:-aurora}"
SEED="${HARDCODED_SEED:-42}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_usage() {
    echo "Polaris Cluster Training Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "🔧 HARDCODED SETTINGS (modify at top of script):"
    echo "  Config file:    ${HARDCODED_CONFIG_FILE:-'Not set'}"
    echo "  Data file:      ${HARDCODED_DATA_FILE:-'Not set'}"
    echo "  Work directory: ${HARDCODED_WORK_DIR:-'Not set'}"
    echo "  Account:        ${HARDCODED_ACCOUNT:-'Not set'}"
    echo "  Nodes:          ${HARDCODED_NODES:-'Not set'}"
    echo "  PPN:            ${HARDCODED_PPN:-'Not set'}"
    echo ""
    echo "Options (override hardcoded settings):"
    echo "  --data PATH          Path to training data file"
    echo "  --config PATH        Path to config file"
    echo "  --work_dir PATH      Working directory"
    echo "  --nodes NUM          Number of nodes"
    echo "  --ppn NUM            Processes per node"
    echo "  --wandb_project NAME Wandb project name"
    echo "  --wandb_name NAME    Wandb run name"
    echo "  --time HOURS         Job time limit in hours"
    echo "  --queue QUEUE        Queue name"
    echo "  --account ACCOUNT    Account/allocation name"
    echo "  --seed NUM           Random seed"
    echo "  --submit             Submit as PBS job instead of interactive run"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Use all hardcoded settings (if configured)"
    echo "  $0 --submit"
    echo ""
    echo "  # Override specific settings"
    echo "  $0 --data /path/to/data.pt --account your_account --submit"
    echo ""
    echo "  # Interactive training (for testing)"
    echo "  $0 --data /path/to/data.pt --account your_account"
    echo ""
    echo "  # Submit as job"
    echo "  $0 --data /path/to/data.jsonl --account your_account --submit --time 8"
    echo ""
}

create_pbs_script() {
    local pbs_file="$WORK_DIR/submit_job.pbs"
    
    cat > "$pbs_file" << EOF
#!/bin/bash
#PBS -N protein_dd
#PBS -l select=${NODES}
#PBS -l filesystems=home:flare
#PBS -l walltime=${TIME_LIMIT}:00:00
#PBS -q ${QUEUE}
#PBS -A ${ACCOUNT}
#PBS -o ${WORK_DIR}/job_output.log
#PBS -e ${WORK_DIR}/job_error.log

module load frameworks/2025.0.0
source /flare/FoundEpidem/avasan/envs/peptide_des_venv/bin/activate
python_path=`which python`
echo $python_path
echo "🌐 Setting environment variables..."
export MPICH_GPU_SUPPORT_ENABLED=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Fix for AF_UNIX path too long error
export TMPDIR="/tmp/pytorch_$$"
mkdir -p $TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR
echo "Set TMPDIR to: $TMPDIR"

# Additional multiprocessing settings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Hang prevention settings
export WANDB_SILENT=true
export WANDB_CONSOLE=off
export HANG_TIMEOUT=900

echo "🚀 Starting interactive training..."
echo "⚠️  Note: For production runs, use --submit to queue the job"
echo ""

cd $PROJECT_ROOT
# Run with MPI
mpiexec -n $((NODES * PPN)) -ppn $PPN \
    python protlig_ddiff/train/run_train_clean.py \
    --config "$CONFIG_FILE" \
    --datafile "$DATA_FILE" \
    --work_dir "$WORK_DIR" \
    --device "$DEVICE" \
    --devicetype "xpu" \
    --cluster "$CLUSTER" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --seed "$SEED" \
    2>&1 | tee "$WORK_DIR/training.log"

echo "Training completed at: \$(date)"

# Cleanup temporary directory
if [[ -n "\$TMPDIR" && "\$TMPDIR" != "/tmp" && -d "\$TMPDIR" ]]; then
    echo "Cleaning up temporary directory: \$TMPDIR"
    rm -rf "\$TMPDIR"
fi

EOF

    echo "📝 Created PBS script: $pbs_file"
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Initialize with hardcoded defaults (can be overridden by command line)
SUBMIT=false

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
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --ppn)
            PPN="$2"
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
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --submit)
            SUBMIT=true
            shift
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
    export AFFINITY_MASK=./running/set_affinity_gpu_polaris.sh
    echo "🌌 Polaris Cluster Protein Discrete Diffusion Training"
    echo "======================================="
    
    # Check required arguments
    if [[ -z "$ACCOUNT" ]]; then
        echo "❌ Account/allocation name is required for Polaris"
        if [[ -n "$HARDCODED_ACCOUNT" ]]; then
            echo "💡 Set HARDCODED_ACCOUNT in the script or use --account your_allocation_name"
        else
            echo "💡 Use --account your_allocation_name or set HARDCODED_ACCOUNT in the script"
        fi
        exit 1
    fi

    if [[ -z "$DATA_FILE" ]]; then
        echo "❌ Data file path is required"
        echo "💡 Set HARDCODED_DATA_FILE in the script or use --data /path/to/data.pt"
        exit 1
    fi

    if [[ ! -f "$DATA_FILE" ]]; then
        echo "❌ Data file not found: $DATA_FILE"
        echo "💡 Check the path and ensure the file exists"
        exit 1
    fi

    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        echo "💡 Check the path or set HARDCODED_CONFIG_FILE in the script"
        exit 1
    fi
    
    # Print configuration
    echo "📊 Final Configuration:"
    echo "   Data file:      $DATA_FILE"
    echo "   Config file:    $CONFIG_FILE"
    echo "   Work directory: $WORK_DIR"
    echo "   Wandb project:  $WANDB_PROJECT"
    echo "   Wandb name:     $WANDB_NAME"
    echo "   Nodes:          $NODES"
    echo "   Processes/node: $PPN"
    echo "   Total ranks:    $((NODES * PPN))"
    echo "   Account:        $ACCOUNT"
    echo "   Time limit:     ${TIME_LIMIT}h"
    echo "   Queue:          $QUEUE"
    echo "   Device:         $DEVICE"
    echo "   Cluster:        $CLUSTER"
    echo "   Seed:           $SEED"
    echo ""
    
    # Create work directory
    mkdir -p "$WORK_DIR"
    echo "📁 Created work directory: $WORK_DIR"
    
    # Copy config for reproducibility
    cp "$CONFIG_FILE" "$WORK_DIR/config.yaml"
    
    if [[ "$SUBMIT" == true ]]; then
        # Create and submit PBS job
        create_pbs_script
        
        echo "🚀 Submitting job to Polaris queue..."
        cd "$(dirname "$WORK_DIR")"
        job_id=$(qsub "$WORK_DIR/submit_job.pbs")
        echo "✅ Job submitted with ID: $job_id"
        echo "📊 Monitor with: qstat $job_id"
        echo "📝 Logs will be in: $WORK_DIR/"
        
    else
        # Interactive run (for testing)
        echo "🔧 Loading Aurora modules..."
        module load frameworks/2025.0.0
        source /flare/FoundEpidem/avasan/envs/peptide_des_venv/bin/activate
        python_path=`which python`
        echo $python_path
        echo "🌐 Setting environment variables..."
        export MPICH_GPU_SUPPORT_ENABLED=1
        export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

        # Fix for AF_UNIX path too long error
        export TMPDIR="/tmp/pytorch_$$"
        mkdir -p $TMPDIR
        export TEMP=$TMPDIR
        export TMP=$TMPDIR
        echo "Set TMPDIR to: $TMPDIR"

        # Additional multiprocessing settings
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1

        # Hang prevention settings
        export WANDB_SILENT=true
        export WANDB_CONSOLE=off
        export HANG_TIMEOUT=900
        
        echo "🚀 Starting interactive training..."
        echo "⚠️  Note: For production runs, use --submit to queue the job"
        echo ""
        
        # Run with MPI
        mpiexec -n $((NODES * PPN)) -ppn $PPN \
            python protlig_ddiff/train/run_train_clean.py \
            --config "$CONFIG_FILE" \
            --datafile "$DATA_FILE" \
            --work_dir "$WORK_DIR" \
            --device "$DEVICE" \
            --devicetype "xpu" \
            --cluster "$CLUSTER" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_name "$WANDB_NAME" \
            --seed "$SEED" \
            2>&1 | tee "$WORK_DIR/training.log"
        #$AFFINITY_MASK \
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            echo "🎉 Training completed successfully!"
        else
            echo "❌ Training failed!"
            exit 1
        fi
    fi
}

# Run main function
main "$@"
