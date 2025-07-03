#!/bin/bash
# GPU Monitoring and Management Script for TransUNet Training

echo "=== GPU Monitoring and Management ==="

# 1. Basic GPU Information
echo "1. GPU Information:"
nvidia-smi

echo -e "\n2. Detailed GPU Status:"
nvidia-smi -l 1  # Updates every 1 second

echo -e "\n3. GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo -e "\n4. GPU Utilization:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv

echo -e "\n5. GPU Temperature:"
nvidia-smi --query-gpu=temperature.gpu --format=csv

echo -e "\n6. Watch GPU in real-time:"
watch -n 1 nvidia-smi

echo -e "\n7. GPU processes:"
nvidia-smi pmon

echo -e "\n8. Kill process by PID (if needed):"
echo "# sudo kill -9 <PID>"
