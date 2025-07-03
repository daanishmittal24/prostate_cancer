# GPU Monitoring & Training Management Guide

## 🐍 **1. Conda Environment Setup**

### Activate Environment
```bash
# Activate your environment
conda activate prostate38

# Check current environment
conda info --envs

# Check which packages are installed
conda list | grep torch
conda list | grep tensorboard
```

### Install Missing Packages
```bash
# Install TensorBoard
conda activate prostate38
pip install tensorboard

# Install monitoring tools
pip install gpustat
pip install wandb  # Advanced logging alternative
pip install psutil  # System monitoring
pip install GPUtil  # GPU monitoring
```

### Quick Activation Setup
Add to your `~/.bashrc` file:
```bash
echo "alias act='conda activate prostate38'" >> ~/.bashrc
echo "alias deact='conda deactivate'" >> ~/.bashrc
source ~/.bashrc
```

## 🖥️ **2. GPU Monitoring Commands**

### Basic GPU Status
```bash
# Simple GPU info
nvidia-smi

# Real-time monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Continuous monitoring
nvidia-smi -l 1
```

### Detailed GPU Monitoring
```bash
# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv

# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv

# GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# All GPU stats
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
```

### Process Monitoring
```bash
# Show processes using GPU
nvidia-smi pmon

# Show detailed process info
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Kill a process (if needed)
sudo kill -9 <PID>
```

### Alternative GPU Tools
```bash
# Install gpustat (better than nvidia-smi)
pip install gpustat

# Use gpustat
gpustat
gpustat -i 1  # Updates every 1 second
```

## 📊 **3. TensorBoard Setup & Alternatives**

### Install & Run TensorBoard
```bash
# Install TensorBoard
conda activate prostate38
pip install tensorboard

# Run TensorBoard
tensorboard --logdir ./logs --host 0.0.0.0 --port 6006

# Access in browser
# http://localhost:6006
# or http://YOUR_SERVER_IP:6006
```

### TensorBoard with SSH Tunneling (Remote Server)
```bash
# On your local machine
ssh -L 6006:localhost:6006 username@server_ip

# Then run tensorboard on server
tensorboard --logdir ./logs

# Access on local browser: http://localhost:6006
```

### Alternative 1: Weights & Biases (wandb)
```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login

# In your training script, replace TensorBoard with:
import wandb
wandb.init(project="prostate-cancer")
wandb.log({"loss": loss, "accuracy": accuracy})
```

### Alternative 2: Simple File Logging
```python
# Use the SimpleLogger class from monitoring_alternatives.py
from monitoring_alternatives import SimpleLogger

logger = SimpleLogger()
logger.log_scalar("train/loss", loss, epoch)
logger.plot_metrics()  # Generate plots
```

## 🚀 **4. Training Management Commands**

### Start Training
```bash
# Activate environment and start training
conda activate prostate38
python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 1

# With background execution
nohup python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 1 > training.log 2>&1 &

# Check background process
ps aux | grep python
```

### Monitor Training
```bash
# Monitor training log
tail -f training.log

# Monitor GPU usage during training
watch -n 1 "nvidia-smi && echo '---' && tail -5 training.log"

# Monitor with gpustat
watch -n 1 "gpustat && echo '---' && tail -3 training.log"
```

### Resume Training
```bash
# Resume from checkpoint
python train_transunet_clean.py \
    --config configs/transunet_config.yaml \
    --resume checkpoints/transunet_best.pth \
    --gpus 1
```

## 📈 **5. Complete Monitoring Setup**

### Terminal 1: Training
```bash
conda activate prostate38
python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 1
```

### Terminal 2: GPU Monitoring
```bash
watch -n 1 nvidia-smi
# or
gpustat -i 1
```

### Terminal 3: TensorBoard (if available)
```bash
conda activate prostate38
tensorboard --logdir ./logs --host 0.0.0.0 --port 6006
```

### Terminal 4: System Monitoring
```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor training logs
tail -f training.log
```

## 🔧 **6. Troubleshooting**

### If TensorBoard Command Not Found
```bash
# Make sure it's installed in the right environment
conda activate prostate38
which tensorboard
pip install tensorboard

# If still not working, use python -m
python -m tensorboard.main --logdir ./logs
```

### If GPU Memory Issues
```bash
# Check what's using GPU memory
nvidia-smi

# Kill processes if needed
sudo kill -9 <PID>

# Reduce batch size in config
# Edit configs/transunet_config.yaml:
# data:
#   batch_size: 8  # Reduce from 16
```

### If Training Stops
```bash
# Check if process is still running
ps aux | grep python

# Check system resources
nvidia-smi
free -h
df -h

# Resume from last checkpoint
python train_transunet_clean.py --resume checkpoints/transunet_best.pth
```

## 📱 **7. Quick Commands Summary**

```bash
# Environment
conda activate prostate38

# Start training
python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 1

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor training logs
tail -f training.log

# Start TensorBoard
tensorboard --logdir ./logs

# Background training
nohup python train_transunet_clean.py --config configs/transunet_config.yaml --gpus 1 > training.log 2>&1 &
```

## 🎯 **8. Best Practices**

1. **Always activate conda environment first**
2. **Monitor GPU memory before starting training**
3. **Use screen/tmux for long training sessions**
4. **Save logs to files for later analysis**
5. **Check disk space regularly**
6. **Use wandb for better visualization if TensorBoard issues persist**
