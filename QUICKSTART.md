# 🚀 RunPod Quick Start for Clearledgr Llama 3.1 8B Fine-tuning

## Prerequisites

### 1. RunPod Account Setup
1. Create account at [RunPod.io](https://runpod.io)
2. Add payment method and credits
3. Request Llama 3.1 access from Meta if needed

### 2. Hugging Face Setup
1. Create account at [HuggingFace.co](https://huggingface.co)
2. Request Llama 3.1 access at [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
3. Generate access token with 'Read' permissions
4. Wait for approval (usually 1-2 hours)

## RunPod Pod Creation

### Recommended Configuration
- **Template**: PyTorch 2.0.1 or newer
- **GPU**: A100 40GB (minimum) or A100 80GB (recommended)
- **Container Disk**: 200GB
- **Expose Ports**: 8888 (Jupyter), 6006 (TensorBoard)
- **Spot Instance**: Enable for cost savings (optional)

### Pod Startup Commands
```bash
# Update system
apt-get update && apt-get install -y git

# Clone your Clearledgr repository
cd /workspace
git clone https://github.com/your-org/clearledgr-2.0.git
cd clearledgr-2.0/runpod_setup

# Set environment variables
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_token_here  # Optional but recommended
```

## One-Command Setup and Training

### Start Complete Workflow
```bash
./scripts/runpod_quickstart.sh start
```

This will automatically:
1. ✅ Check GPU and environment
2. ✅ Install all dependencies
3. ✅ Setup Hugging Face authentication
4. ✅ Download Llama 3.1 8B model
5. ✅ Prepare Clearledgr financial dataset
6. ✅ Start fine-tuning with LoRA
7. ✅ Launch real-time monitoring

### Monitor Training Progress
```bash
# Check status
./scripts/runpod_quickstart.sh status

# Follow training logs
./scripts/runpod_quickstart.sh logs

# Interactive monitoring
./scripts/runpod_quickstart.sh monitor
```

### Stop Training
```bash
./scripts/runpod_quickstart.sh stop
```

## Manual Step-by-Step Process

If you prefer manual control:

### 1. Install Dependencies
```bash
pip install -r requirements-runpod.txt
```

### 2. Setup Authentication
```bash
huggingface-cli login
# Enter your HF token when prompted
```

### 3. Download Model
```bash
python scripts/download_llama_runpod.py --model-size 8B
```

### 4. Prepare Dataset
```bash
python scripts/prepare_financial_dataset.py
```

### 5. Start Training
```bash
python scripts/finetune_llama_runpod.py \
  --model-path ./models/llama-3.1-8b \
  --dataset-path ./data/financial_training_data/clearledgr_financial_dataset \
  --output-dir ./models/clearledgr-llama-3.1-8b \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --num-epochs 3 \
  --use-4bit \
  --use-lora
```

### 6. Monitor Training
```bash
python scripts/monitor_training.py --log-path logs/training.log
```

## Expected Training Time and Cost

### Training Duration
- **A100 40GB**: 18-24 hours
- **A100 80GB**: 12-16 hours
- **H100**: 8-12 hours

### Cost Estimation (Spot Pricing)
- **A100 40GB**: ~$45-60 total
- **A100 80GB**: ~$35-50 total
- **H100**: ~$65-85 total

## Monitoring and Outputs

### Real-time Monitoring
- **GPU Utilization**: nvidia-smi updates
- **Training Metrics**: Loss, learning rate, epoch progress
- **System Resources**: CPU, RAM, disk usage
- **Plots**: Auto-generated training curves

### Output Files
```
models/clearledgr-llama-3.1-8b/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── training_config.json         # Training parameters
├── tokenizer.json              # Tokenizer files
└── training_args.bin           # Training arguments

logs/
├── training.log                # Full training log
├── training.pid               # Process ID
└── monitor.pid               # Monitor process ID

monitoring_output/
├── monitoring_data.json       # System metrics
├── training_loss.png         # Training curves
└── gpu_utilization.png       # GPU usage plots
```

## Troubleshooting

### Common Issues

**OOM (Out of Memory) Errors**
```bash
# Reduce batch size
python scripts/finetune_llama_runpod.py --batch-size 2 --gradient-accumulation-steps 8
```

**Slow Training**
```bash
# Check GPU utilization
nvidia-smi

# Verify dataset loading
python -c "from datasets import load_from_disk; ds = load_from_disk('./data/financial_training_data/clearledgr_financial_dataset'); print(len(ds))"
```

**Authentication Issues**
```bash
# Re-login to Hugging Face
huggingface-cli login --token $HF_TOKEN

# Verify access to Llama
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')"
```

### Getting Help

1. **Check logs**: `tail -f logs/training.log`
2. **Monitor resources**: `./scripts/runpod_quickstart.sh monitor`
3. **RunPod support**: Use RunPod community Discord
4. **Hugging Face issues**: Check model access status

## Post-Training Steps

### 1. Model Validation
```bash
python scripts/test_fine_tuned_model.py --model-path ./models/clearledgr-llama-3.1-8b
```

### 2. Model Export
```bash
# Export for deployment
python scripts/export_model.py --input-dir ./models/clearledgr-llama-3.1-8b --output-format huggingface
```

### 3. Download Results
```bash
# Compress model for download
tar -czf clearledgr-llama-3.1-8b.tar.gz ./models/clearledgr-llama-3.1-8b/

# Use RunPod's file manager or scp to download
```

## Integration with Clearledgr

Once training is complete, integrate the fine-tuned model:

1. **Update HybridModelRouter** to use the new model
2. **Deploy to inference infrastructure**
3. **A/B test** against base model
4. **Monitor performance** on real financial tasks

## Security Notes

- ✅ All training happens in isolated RunPod environment
- ✅ No customer data used in training (synthetic examples only)
- ✅ Model weights encrypted at rest
- ✅ Access via SSH keys only
- ✅ No persistent data after pod termination

## Next Steps

After successful fine-tuning:
1. **Test model quality** on financial benchmarks
2. **Integrate with Clearledgr agents**
3. **Deploy to production inference**
4. **Monitor real-world performance**
5. **Plan next iteration** based on results

---

**Ready to fine-tune Llama 3.1 8B for Clearledgr's AI finance team!** 🚀
