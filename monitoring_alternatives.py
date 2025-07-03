#!/usr/bin/env python3
"""
Alternative monitoring solutions for TransUNet training
Includes Weights & Biases (wandb) integration and simple logging
"""

import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime

class SimpleLogger:
    """Simple file-based logging alternative to TensorBoard"""
    
    def __init__(self, log_dir="./logs", experiment_name="transunet"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_log.json")
        self.metrics = {}
        
        os.makedirs(log_dir, exist_ok=True)
        
    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if tag not in self.metrics:
            self.metrics[tag] = []
        
        self.metrics[tag].append({
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, save_dir="./plots"):
        """Generate plots of metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        for metric_name, data in self.metrics.items():
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values)
            plt.title(f'{metric_name} over Training')
            plt.xlabel('Step/Epoch')
            plt.ylabel(metric_name)
            plt.grid(True)
            
            plot_path = os.path.join(save_dir, f'{metric_name.replace("/", "_")}.png')
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Plot saved: {plot_path}")

def setup_wandb_logging():
    """Setup Weights & Biases logging"""
    try:
        import wandb
        
        # Initialize wandb
        wandb.init(
            project="prostate-cancer-transunet",
            name="transunet-training",
            config={
                "model": "TransUNet",
                "dataset": "PANDA",
                "architecture": "ViT + U-Net",
                "epochs": 25,
                "batch_size": 16,
                "learning_rate": 1e-4
            }
        )
        
        print("✅ Weights & Biases logging initialized")
        return wandb
        
    except ImportError:
        print("❌ wandb not installed. Install with: pip install wandb")
        return None

def log_system_metrics():
    """Log system metrics (GPU, CPU, Memory)"""
    try:
        import psutil
        import GPUtil
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_metrics = {
                'gpu_utilization': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature
            }
        else:
            gpu_metrics = {}
        
        # CPU and RAM metrics
        cpu_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        return {**gpu_metrics, **cpu_metrics}
        
    except ImportError:
        print("Install monitoring packages: pip install psutil GPUtil")
        return {}

if __name__ == "__main__":
    print("=== Alternative Monitoring Setup ===")
    
    # Test simple logger
    logger = SimpleLogger()
    logger.log_scalar("test/loss", 2.5, 0)
    logger.log_scalar("test/accuracy", 0.3, 0)
    print("✅ Simple logger working")
    
    # Test wandb
    wandb_logger = setup_wandb_logging()
    
    # Test system metrics
    metrics = log_system_metrics()
    if metrics:
        print("✅ System metrics available:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print("\n=== Usage Instructions ===")
    print("1. For simple logging: Use SimpleLogger class in training script")
    print("2. For wandb: Install with 'pip install wandb' and use wandb.log()")
    print("3. For system monitoring: Use log_system_metrics() function")
