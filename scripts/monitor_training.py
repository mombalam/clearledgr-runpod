#!/usr/bin/env python3
"""
Monitor Llama 3.1 8B Training Progress on RunPod

Real-time monitoring script for tracking training metrics, GPU usage,
and system resources during fine-tuning.
"""

import os
import time
import psutil
import GPUtil
import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def get_gpu_stats():
    """Get GPU utilization and memory stats"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "gpu_utilization": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return None

def get_system_stats():
    """Get system CPU and memory stats"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "disk_used_percent": psutil.disk_usage('/').percent
    }

def parse_training_log(log_path):
    """Parse training log for metrics"""
    if not os.path.exists(log_path):
        return None
    
    metrics = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if '"train_loss"' in line or '"eval_loss"' in line:
                    try:
                        # Try to parse JSON log line
                        data = json.loads(line.strip())
                        metrics.append(data)
                    except:
                        continue
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return metrics

def create_training_plots(metrics, output_dir):
    """Create training progress plots"""
    if not metrics:
        return
    
    df = pd.DataFrame(metrics)
    
    # Training loss plot
    if 'train_loss' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['train_loss'], label='Training Loss')
        
        if 'eval_loss' in df.columns:
            eval_df = df.dropna(subset=['eval_loss'])
            plt.plot(eval_df['step'], eval_df['eval_loss'], label='Validation Loss')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Clearledgr Llama 3.1 8B Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
        plt.close()

def monitor_training(args):
    """Main monitoring function"""
    print("🔍 Starting Clearledgr Llama 3.1 8B Training Monitor")
    print(f"Log path: {args.log_path}")
    print(f"Update interval: {args.interval} seconds")
    print("=" * 60)
    
    start_time = datetime.now()
    iteration = 0
    
    # Create monitoring output directory
    monitor_dir = Path("monitoring_output")
    monitor_dir.mkdir(exist_ok=True)
    
    # Store monitoring data
    monitoring_data = []
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now()
            
            # Get system stats
            gpu_stats = get_gpu_stats()
            system_stats = get_system_stats()
            
            # Parse training log
            training_metrics = parse_training_log(args.log_path)
            
            # Current status
            print(f"\n⏰ Update #{iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Runtime: {current_time - start_time}")
            
            # GPU info
            if gpu_stats:
                print(f"🎮 GPU: {gpu_stats['gpu_utilization']:.1f}% utilization")
                print(f"💾 VRAM: {gpu_stats['memory_used']:.0f}MB / {gpu_stats['memory_total']:.0f}MB ({gpu_stats['memory_percent']:.1f}%)")
                print(f"🌡️  Temp: {gpu_stats['temperature']:.0f}°C")
            
            # System info
            print(f"🖥️  CPU: {system_stats['cpu_percent']:.1f}%")
            print(f"💽 RAM: {system_stats['memory_used_gb']:.1f}GB / {system_stats['memory_total_gb']:.1f}GB ({system_stats['memory_percent']:.1f}%)")
            print(f"💿 Disk: {system_stats['disk_used_percent']:.1f}%")
            
            # Training progress
            if training_metrics:
                latest_metric = training_metrics[-1]
                print(f"📈 Training Step: {latest_metric.get('step', 'N/A')}")
                print(f"📉 Train Loss: {latest_metric.get('train_loss', 'N/A'):.4f}" if 'train_loss' in latest_metric else "")
                print(f"📊 Eval Loss: {latest_metric.get('eval_loss', 'N/A'):.4f}" if 'eval_loss' in latest_metric else "")
                print(f"⚡ Learning Rate: {latest_metric.get('learning_rate', 'N/A')}")
            
            # Store data for analysis
            monitoring_data.append({
                "timestamp": current_time.isoformat(),
                "gpu_stats": gpu_stats,
                "system_stats": system_stats,
                "training_metrics": training_metrics[-1] if training_metrics else None
            })
            
            # Save monitoring data periodically
            if iteration % 10 == 0:
                with open(monitor_dir / "monitoring_data.json", 'w') as f:
                    json.dump(monitoring_data, f, indent=2)
                
                # Create plots
                if training_metrics:
                    create_training_plots(training_metrics, monitor_dir)
            
            # Check for completion
            if args.auto_stop and training_metrics:
                latest = training_metrics[-1]
                if 'epoch' in latest and latest['epoch'] >= 3:
                    print("\n🎉 Training appears to be completed!")
                    break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
    
    finally:
        # Save final monitoring data
        with open(monitor_dir / "monitoring_data.json", 'w') as f:
            json.dump(monitoring_data, f, indent=2)
        
        print(f"\n📁 Monitoring data saved to: {monitor_dir}")
        print("👋 Monitoring ended")

def main():
    parser = argparse.ArgumentParser(description="Monitor Llama 3.1 8B training on RunPod")
    parser.add_argument("--log-path", default="logs/training.log", 
                       help="Path to training log file")
    parser.add_argument("--interval", type=int, default=30,
                       help="Update interval in seconds")
    parser.add_argument("--auto-stop", action="store_true",
                       help="Auto-stop monitoring when training completes")
    parser.add_argument("--output-dir", default="monitoring_output",
                       help="Directory for monitoring outputs")
    
    args = parser.parse_args()
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    
    monitor_training(args)

if __name__ == "__main__":
    main()
