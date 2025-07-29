#!/usr/bin/env python3
"""
Download Llama 3.1 8B Model for RunPod Fine-tuning

Optimized download script for RunPod environment with progress monitoring
and checkpoint resumption capabilities.
"""

import os
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, login
import psutil
import GPUtil

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_system_resources():
    """Check available system resources"""
    logger = logging.getLogger(__name__)
    
    # Check available disk space
    disk_usage = psutil.disk_usage('/')
    available_gb = disk_usage.free / (1024**3)
    
    logger.info(f"Available disk space: {available_gb:.1f} GB")
    
    if available_gb < 50:
        logger.warning("Low disk space! Llama 3.1 8B requires ~30GB")
        
    # Check GPU memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            logger.info(f"GPU: {gpu.name}, Memory: {gpu.memoryTotal}MB")
            if gpu.memoryTotal < 20000:  # 20GB minimum
                logger.warning("GPU memory may be insufficient for 8B model fine-tuning")
        else:
            logger.warning("No GPU detected!")
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")

def download_llama_model(model_name: str, local_dir: str, token: str = None):
    """Download Llama model with progress monitoring"""
    logger = logging.getLogger(__name__)
    
    # Authenticate with Hugging Face
    if token:
        login(token=token)
    else:
        # Try to use existing authentication
        try:
            login()
        except Exception as e:
            logger.error("Hugging Face authentication failed. Please run 'huggingface-cli login' first")
            return False
    
    logger.info(f"Starting download of {model_name}")
    logger.info(f"Download directory: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download model with progress
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Important for RunPod
            resume_download=True,
            token=token
        )
        
        logger.info("✅ Model download completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def verify_download(model_dir: str):
    """Verify the downloaded model is complete"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    model_path = Path(model_dir)
    
    logger.info("Verifying download...")
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    
    # Check for model shards
    safetensors_files = list(model_path.glob("model-*.safetensors"))
    if not safetensors_files:
        logger.warning("No model safetensors files found")
        return False
    
    logger.info(f"✅ Model verification passed! Found {len(safetensors_files)} model files")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Llama 3.1 8B for RunPod fine-tuning")
    parser.add_argument("--model-size", default="8B", choices=["8B", "70B"], 
                       help="Model size to download")
    parser.add_argument("--output-dir", default="./models", 
                       help="Directory to save the model")
    parser.add_argument("--token", help="Hugging Face token (optional if already logged in)")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify download after completion")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Model configuration
    model_configs = {
        "8B": "meta-llama/Llama-3.1-8B",
        "70B": "meta-llama/Llama-3.1-70B"
    }
    
    model_name = model_configs[args.model_size]
    model_dir = os.path.join(args.output_dir, f"llama-3.1-{args.model_size.lower()}")
    
    logger.info("🚀 Starting Llama 3.1 8B download for Clearledgr fine-tuning")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {model_dir}")
    
    # Check system resources
    check_system_resources()
    
    # Download model
    success = download_llama_model(model_name, model_dir, args.token)
    
    if success and args.verify:
        verify_download(model_dir)
    
    if success:
        logger.info("🎉 Download complete! Ready for fine-tuning.")
        logger.info(f"Model location: {model_dir}")
        logger.info("Next step: python scripts/finetune_llama_runpod.py")
    else:
        logger.error("❌ Download failed. Please check your authentication and try again.")

if __name__ == "__main__":
    main()
