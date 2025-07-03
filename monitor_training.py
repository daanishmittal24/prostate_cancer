#!/usr/bin/env python3
"""
Simple Training Monitor for TransUNet
Monitors GPU usage, training progress, and logs
"""

import os
import time
import json
import subprocess
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        # Get GPU memory usage
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            memory_used, memory_total, gpu_util, temp = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(memory_used),
                'memory_total_mb': int(memory_total),
                'memory_percent': round(int(memory_used) / int(memory_total) * 100, 1),
                'gpu_utilization': int(gpu_util),
                'temperature': int(temp)
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    
    return None

def get_training_progress():
    """Parse training log for progress"""
    log_file = "training.log"
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find last epoch info
        for line in reversed(lines):
            if "Train - Loss:" in line:
                # Parse: Train - Loss: 1.2345, Acc: 0.6789, Kappa: 0.1234
                parts = line.strip().split(", ")
                train_loss = float(parts[0].split(": ")[1])
                train_acc = float(parts[1].split(": ")[1])
                train_kappa = float(parts[2].split(": ")[1])
                return {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_kappa': train_kappa,
                    'last_update': datetime.now().strftime("%H:%M:%S")
                }
    except Exception as e:
        print(f"Error parsing training log: {e}")
    
    return None

def print_status():
    """Print current training status"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("🚀 TransUNet Training Monitor")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # GPU Status
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print("🖥️  GPU Status:")
        print(f"   Memory: {gpu_stats['memory_used_mb']}MB / {gpu_stats['memory_total_mb']}MB ({gpu_stats['memory_percent']}%)")
        print(f"   Utilization: {gpu_stats['gpu_utilization']}%")
        print(f"   Temperature: {gpu_stats['temperature']}°C")
    else:
        print("❌ GPU Status: Not available")
    
    print()
    
    # Training Progress
    progress = get_training_progress()
    if progress:
        print("📈 Training Progress:")
        print(f"   Loss: {progress['train_loss']:.4f}")
        print(f"   Accuracy: {progress['train_acc']:.4f}")
        print(f"   Kappa: {progress['train_kappa']:.4f}")
        print(f"   Last Update: {progress['last_update']}")
    else:
        print("⏳ Training Progress: Starting or no log file")
    
    print()
    
    # Check if training process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'train_transunet'], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training Process: Running")
        else:
            print("❌ Training Process: Not detected")
    except:
        print("❓ Training Process: Cannot check")
    
    print()
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)

def main():
    """Main monitoring loop"""
    print("Starting TransUNet Training Monitor...")
    
    try:
        while True:
            print_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()
