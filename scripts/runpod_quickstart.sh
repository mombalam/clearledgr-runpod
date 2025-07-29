#!/bin/bash

# Clearledgr Llama 3.1 8B RunPod Quick Start Script
# Automates the complete fine-tuning workflow

set -e

echo "🚀 Clearledgr Llama 3.1 8B Fine-tuning on RunPod"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_SIZE="8B"
OUTPUT_DIR="./models/clearledgr-llama-3.1-8b"
DATASET_PATH="./data/financial_training_data/clearledgr_financial_dataset"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're on RunPod
check_runpod_environment() {
    if [ -z "$RUNPOD_POD_ID" ]; then
        print_warning "RUNPOD_POD_ID not found. Are you running on RunPod?"
    else
        print_status "Running on RunPod Pod: $RUNPOD_POD_ID"
    fi
}

# Check GPU availability
check_gpu() {
    print_step "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        print_status "GPU detected and available"
    else
        print_error "nvidia-smi not found. GPU not available!"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ ! -f "requirements-runpod.txt" ]; then
        print_error "requirements-runpod.txt not found!"
        exit 1
    fi
    
    pip install -r requirements-runpod.txt
    print_status "Dependencies installed successfully"
}

# Setup Hugging Face authentication
setup_hf_auth() {
    print_step "Setting up Hugging Face authentication..."
    
    if [ -z "$HF_TOKEN" ]; then
        print_warning "HF_TOKEN environment variable not set"
        echo "Please run: export HF_TOKEN=your_huggingface_token"
        echo "Or run: huggingface-cli login"
        read -p "Press enter after setting up authentication..."
    else
        echo $HF_TOKEN | huggingface-cli login --token
        print_status "Hugging Face authentication configured"
    fi
}

# Create necessary directories
setup_directories() {
    print_step "Creating directories..."
    
    mkdir -p logs
    mkdir -p models
    mkdir -p data/financial_training_data
    mkdir -p monitoring_output
    
    print_status "Directories created"
}

# Download Llama model
download_model() {
    print_step "Downloading Llama 3.1 ${MODEL_SIZE} model..."
    
    if [ -d "./models/llama-3.1-8b" ]; then
        print_warning "Model directory already exists. Skipping download..."
        return
    fi
    
    python scripts/download_llama_runpod.py \
        --model-size $MODEL_SIZE \
        --output-dir ./models \
        --verify
    
    print_status "Model download completed"
}

# Prepare training dataset
prepare_dataset() {
    print_step "Preparing Clearledgr financial training dataset..."
    
    if [ -d "$DATASET_PATH" ]; then
        print_warning "Dataset already exists. Skipping preparation..."
        return
    fi
    
    python scripts/prepare_financial_dataset.py
    
    print_status "Dataset preparation completed"
}

# Start training
start_training() {
    print_step "Starting Llama 3.1 8B fine-tuning..."
    
    # Set up W&B if token is available
    if [ ! -z "$WANDB_API_KEY" ]; then
        print_status "Weights & Biases tracking enabled"
    else
        print_warning "WANDB_API_KEY not set. Training will proceed without W&B tracking"
    fi
    
    # Start training in background
    nohup python scripts/finetune_llama_runpod.py \
        --model-path ./models/llama-3.1-8b \
        --dataset-path $DATASET_PATH \
        --output-dir $OUTPUT_DIR \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
        --learning-rate 2e-5 \
        --num-epochs 3 \
        --save-steps 500 \
        --eval-steps 500 \
        --logging-steps 100 \
        --use-4bit \
        --use-lora > logs/training.log 2>&1 &
    
    TRAINING_PID=$!
    echo $TRAINING_PID > logs/training.pid
    
    print_status "Training started with PID: $TRAINING_PID"
    print_status "Training log: logs/training.log"
}

# Start monitoring
start_monitoring() {
    print_step "Starting training monitor..."
    
    sleep 5  # Give training a moment to start
    
    python scripts/monitor_training.py \
        --log-path logs/training.log \
        --interval 30 \
        --auto-stop &
    
    MONITOR_PID=$!
    echo $MONITOR_PID > logs/monitor.pid
    
    print_status "Monitor started with PID: $MONITOR_PID"
}

# Show status
show_status() {
    echo ""
    echo "🎯 Training Status"
    echo "=================="
    
    if [ -f "logs/training.pid" ]; then
        TRAINING_PID=$(cat logs/training.pid)
        if kill -0 $TRAINING_PID 2>/dev/null; then
            print_status "Training is running (PID: $TRAINING_PID)"
        else
            print_warning "Training process not found"
        fi
    fi
    
    if [ -f "logs/monitor.pid" ]; then
        MONITOR_PID=$(cat logs/monitor.pid)
        if kill -0 $MONITOR_PID 2>/dev/null; then
            print_status "Monitor is running (PID: $MONITOR_PID)"
        else
            print_warning "Monitor process not found"
        fi
    fi
    
    echo ""
    echo "📁 Important Files:"
    echo "   Training log: logs/training.log"
    echo "   Monitor data: monitoring_output/"
    echo "   Model output: $OUTPUT_DIR"
    
    echo ""
    echo "📊 Useful Commands:"
    echo "   tail -f logs/training.log     # Follow training log"
    echo "   nvidia-smi                    # Check GPU usage"
    echo "   python scripts/monitor_training.py  # Manual monitoring"
}

# Cleanup function
cleanup() {
    print_step "Cleaning up processes..."
    
    if [ -f "logs/training.pid" ]; then
        TRAINING_PID=$(cat logs/training.pid)
        if kill -0 $TRAINING_PID 2>/dev/null; then
            kill $TRAINING_PID
            print_status "Training process stopped"
        fi
    fi
    
    if [ -f "logs/monitor.pid" ]; then
        MONITOR_PID=$(cat logs/monitor.pid)
        if kill -0 $MONITOR_PID 2>/dev/null; then
            kill $MONITOR_PID
            print_status "Monitor process stopped"
        fi
    fi
}

# Main execution
main() {
    case "${1:-start}" in
        "start")
            check_runpod_environment
            check_gpu
            setup_directories
            install_dependencies
            setup_hf_auth
            download_model
            prepare_dataset
            start_training
            start_monitoring
            show_status
            ;;
        "status")
            show_status
            ;;
        "stop")
            cleanup
            ;;
        "monitor")
            python scripts/monitor_training.py
            ;;
        "logs")
            tail -f logs/training.log
            ;;
        *)
            echo "Usage: $0 {start|status|stop|monitor|logs}"
            echo ""
            echo "Commands:"
            echo "  start   - Start complete fine-tuning workflow"
            echo "  status  - Show current training status"
            echo "  stop    - Stop all training processes"
            echo "  monitor - Start interactive monitoring"
            echo "  logs    - Follow training logs"
            exit 1
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"
