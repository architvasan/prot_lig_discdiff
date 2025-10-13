#!/bin/bash

# Polaris Cluster Training Script for Protein Discrete Diffusion
# Optimized for Intel XPU with MPI support

set -e

# =============================================================================
# AURORA-SPECIFIC CONFIGURATION
# =============================================================================

# Default Polaris settings
PWD="/eagle/FoundEpidem/avasan/IDEAL/DiffusionModels/prot_lig_discdiff"
CONFIG_FILE="${PWD}/config_protein.yaml"
DATA_FILE="${PWD}/input_data/processed_uniref50.pt"  # CHANGE THIS
WORK_DIR="${PWD}/experiments/sedd_$(date +%Y%m%d_%H%M%S)"  # CHANGE PROJECT PATH

# Polaris-specific defaults
DEVICE="cuda:0"
CLUSTER="polaris"
WANDB_PROJECT="protein-discrete-diffusion-aurora"
NODES=1
PPN=4  # Processes per node (Polaris has 4 GPUs per node)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_usage() {
    echo "Polaris Cluster Training Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data PATH          Path to training data file (required)"
    echo "  --config PATH        Path to config file (default: $CONFIG_FILE)"
    echo "  --work_dir PATH      Working directory (default: auto-generated)"
    echo "  --nodes NUM          Number of nodes (default: $NODES)"
    echo "  --ppn NUM            Processes per node (default: $PPN)"
    echo "  --wandb_project NAME Wandb project name (default: $WANDB_PROJECT)"
    echo "  --wandb_name NAME    Wandb run name (default: auto-generated)"
    echo "  --time HOURS         Job time limit in hours (default: 2)"
    echo "  --queue QUEUE        Queue name (default: workq)"
    echo "  --account ACCOUNT    Account/allocation name (required for submission)"
    echo "  --submit             Submit as PBS job instead of interactive run"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Interactive training (for testing)"
    echo "  $0 --data /path/to/data.jsonl --account your_account"
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
#PBS -l select=${NODES}:system=aurora
#PBS -l place=scatter
#PBS -l walltime=${TIME_LIMIT}:00:00
#PBS -q ${QUEUE}
#PBS -A ${ACCOUNT}
#PBS -o ${WORK_DIR}/job_output.log
#PBS -e ${WORK_DIR}/job_error.log

# Load Polaris modules
module use /soft/modulefiles
module load conda

# Set environment variables
#export MPICH_GPU_SUPPORT_ENABLED=1
#export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Change to work directory
cd \$PBS_O_WORKDIR

# Run training with MPI
mpiexec -n \$((${NODES} * ${PPN})) -ppn ${PPN} \\
    python run_train_clean.py \\
    --config ${CONFIG_FILE} \\
    --datafile ${DATA_FILE} \\
    --work_dir ${WORK_DIR} \\
    --device xpu:0 \\
    --cluster aurora \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${WANDB_NAME} \\
    --seed 42

echo "Training completed at: \$(date)"
EOF

    echo "📝 Created PBS script: $pbs_file"
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

WANDB_NAME="aurora-disc-diff-$(date +%Y%m%d_%H%M%S)"
TIME_LIMIT=2
QUEUE="workq"
ACCOUNT=""
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
    export AFFINITY_MASK=./set_affinity_gpu_polaris.sh
    echo "🌌 Polaris Cluster Protein Discrete Diffusion Training"
    echo "======================================="
    
    # Check required arguments
    if [[ -z "$ACCOUNT" ]]; then
        echo "❌ Account/allocation name is required for Polaris"
        echo "💡 Use --account your_allocation_name"
        exit 1
    fi
    
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "❌ Data file not found: $DATA_FILE"
        echo "💡 Update the DATA_FILE path in this script or use --data"
        exit 1
    fi
    
    # Print configuration
    echo "📊 Configuration:"
    echo "   Data file:      $DATA_FILE"
    echo "   Config file:    $CONFIG_FILE"
    echo "   Work directory: $WORK_DIR"
    echo "   Nodes:          $NODES"
    echo "   Processes/node: $PPN"
    echo "   Total ranks:    $((NODES * PPN))"
    echo "   Account:        $ACCOUNT"
    echo "   Time limit:     ${TIME_LIMIT}h"
    echo "   Queue:          $QUEUE"
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
        echo "🔧 Loading Polaris modules..."
        module use /soft/modulefiles 2>/dev/null || true
        module load conda/2025-09-25 2>/dev/null || true
        conda activate
        source ../protein_lig_sedd/prot_lig_sedd/bin/activate 
        python_path=`which python`
        echo $python_path
        echo "🌐 Setting environment variables..."
        #export MPICH_GPU_SUPPORT_ENABLED=1
        #export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
        
        echo "🚀 Starting interactive training..."
        echo "⚠️  Note: For production runs, use --submit to queue the job"
        echo ""
        
        # Run with MPI
        mpiexec -n $((NODES * PPN)) -ppn $PPN \
            $AFFINITY_MASK \
            python protlig_ddiff/train/run_train_clean.py \
            --config "$CONFIG_FILE" \
            --datafile "$DATA_FILE" \
            --work_dir "$WORK_DIR" \
            --device cuda:0 \
            --cluster polaris \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_name "$WANDB_NAME" \
            --seed 42 \
            2>&1 | tee "$WORK_DIR/training.log"
        
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
