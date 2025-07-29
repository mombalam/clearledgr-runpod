# RunPod Setup for Clearledgr Llama 3.1 8B Fine-tuning

## Overview

This setup configures RunPod for fine-tuning Llama 3.1 8B specifically for Clearledgr's AI finance team. The fine-tuned model will power our financial agents with domain-specific knowledge.

## RunPod Configuration

### Required GPU Configuration
- **GPU**: A100 40GB or A100 80GB (recommended for 8B model)
- **VRAM**: Minimum 24GB, recommended 40GB+
- **Storage**: 200GB+ for model, datasets, and checkpoints
- **Template**: PyTorch 2.0+ with CUDA 11.8+

### Container Requirements
```bash
# Base image with PyTorch and transformers
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Additional packages will be installed via requirements.txt
```

## Setup Steps

### 1. RunPod Pod Creation
1. Go to RunPod.io and create account
2. Select "GPU Pods" → "Deploy"
3. Choose template: "RunPod PyTorch 2.0"
4. GPU: A100 40GB (minimum) or A100 80GB (recommended)
5. Container Disk: 200GB
6. Expose ports: 8888 (Jupyter), 6006 (TensorBoard)

### 2. Initial Environment Setup
```bash
# Connect to your pod via SSH or web terminal
# Install required packages
pip install -r requirements-training.txt

# Set up Hugging Face authentication
huggingface-cli login
# Enter your HF token with Llama access
```

### 3. Download and Prepare Llama 3.1 8B
```bash
# Download the base model
python scripts/download_llama_runpod.py --model-size 8B

# Verify model download
python scripts/verify_model.py --model-path ./models/llama-3.1-8b
```

### 4. Prepare Financial Training Data
```bash
# Process Clearledgr financial dataset
python scripts/prepare_financial_dataset.py

# Validate dataset format
python scripts/validate_training_data.py
```

### 5. Start Fine-tuning
```bash
# Launch fine-tuning with optimal RunPod settings
python scripts/finetune_llama_runpod.py \
  --model-name meta-llama/Llama-3.1-8B \
  --dataset-path ./data/clearledgr_financial_dataset \
  --output-dir ./models/clearledgr-llama-3.1-8b \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --num-epochs 3 \
  --save-steps 500 \
  --eval-steps 500 \
  --logging-steps 100
```

## Cost Optimization

### RunPod Pricing (Approximate)
- **A100 40GB**: ~$1.89/hour
- **A100 80GB**: ~$2.89/hour
- **Training Duration**: 12-24 hours for full fine-tune
- **Estimated Cost**: $23-70 for complete training

### Cost-Saving Tips
1. Use spot instances when available (50-70% discount)
2. Monitor training progress and stop early if converged
3. Use gradient checkpointing to reduce memory usage
4. Consider LoRA fine-tuning for faster/cheaper training

## Monitoring and Management

### Training Monitoring
- **TensorBoard**: http://[pod-ip]:6006
- **Training Logs**: Real-time via terminal
- **Model Checkpoints**: Saved every 500 steps
- **Evaluation Metrics**: Loss, perplexity, financial accuracy

### Resource Monitoring
```bash
# GPU utilization
nvidia-smi

# Memory usage
watch -n 1 'free -h && df -h'

# Training progress
tail -f logs/training.log
```

## Security and Data Protection

### Model Security
- All training happens in isolated RunPod environment
- Models stored in encrypted storage
- Access via SSH keys only
- No persistent data on pod after training

### Data Privacy
- Financial training data encrypted at rest
- No customer data used in training
- Synthetic financial scenarios only
- GDPR/SOX compliant training process

## Next Steps After Training

1. **Model Validation**: Test on financial benchmarks
2. **Integration**: Deploy to Clearledgr inference infrastructure
3. **A/B Testing**: Compare with base model performance
4. **Agent Integration**: Update HybridModelRouter
5. **Performance Monitoring**: Track financial accuracy improvements

## Troubleshooting

### Common Issues
- **OOM Errors**: Reduce batch size or enable gradient checkpointing
- **Slow Training**: Check GPU utilization and data loading
- **Connection Issues**: Verify RunPod network configuration
- **Authentication**: Ensure HuggingFace token has Llama access

### Support Resources
- RunPod Documentation: https://docs.runpod.io/
- Llama Fine-tuning Guide: See `/scripts/training_guide.md`
- Clearledgr Training Issues: Contact AI team

## Files in This Setup

```
runpod_setup/
├── README.md                     # This file
├── requirements-runpod.txt       # RunPod-specific dependencies
├── scripts/
│   ├── download_llama_runpod.py  # Optimized model download
│   ├── finetune_llama_runpod.py  # RunPod fine-tuning script
│   ├── prepare_financial_dataset.py
│   ├── verify_model.py
│   └── monitor_training.py
├── configs/
│   ├── runpod_training_config.yaml
│   └── lora_config.yaml
└── data/
    └── financial_training_samples/
```
