@echo off
REM ViTDet + Mask R-CNN Setup and Training Script for Windows
REM This script provides a simple interface for setting up and running the training

echo ViTDet + Mask R-CNN Training Setup for Windows
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check for GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Warning: No NVIDIA GPU detected or nvidia-smi not available
    set /p continue="Continue without GPU? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
)

REM Get data directory from user
set /p DATA_DIR="Enter path to your data directory: "
if not exist "%DATA_DIR%" (
    echo Error: Data directory not found: %DATA_DIR%
    pause
    exit /b 1
)

REM Check data structure
echo Checking data directory structure...
if not exist "%DATA_DIR%\train.csv" (
    echo Error: train.csv not found in data directory
    goto :data_error
)
if not exist "%DATA_DIR%\train_images" (
    echo Error: train_images directory not found
    goto :data_error
)
if not exist "%DATA_DIR%\train_label_masks" (
    echo Error: train_label_masks directory not found
    goto :data_error
)

echo Data directory structure is correct!

REM Ask about installation
set /p INSTALL="Install/update dependencies? (Y/n): "
if /i "%INSTALL%"=="n" goto :skip_install

echo Installing dependencies...
echo This may take several minutes...

REM Install PyTorch with CUDA
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install Detectron2
echo Installing Detectron2...
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

REM Install other dependencies
echo Installing other dependencies...
if exist requirements_detectron.txt (
    pip install -r requirements_detectron.txt
) else (
    pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 albumentations>=1.2.0
    pip install opencv-python>=4.5.0 Pillow>=8.0.0 matplotlib>=3.5.0 seaborn>=0.11.0
    pip install tensorboard>=2.8.0 wandb>=0.12.0 tqdm>=4.62.0 PyYAML>=6.0 pycocotools>=2.0.4
)

echo Verifying installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

:skip_install

REM Get number of GPUs
set /p NUM_GPUS="Number of GPUs to use (default 4): "
if "%NUM_GPUS%"=="" set NUM_GPUS=4

REM Update config
echo Updating configuration...
python -c "import yaml; config=yaml.safe_load(open('configs/vitdet_config.yaml')); config['data']['data_dir']='%DATA_DIR%'; yaml.dump(config, open('configs/vitdet_config.yaml', 'w'))"

REM Launch training
echo Starting training...
echo Configuration: configs/vitdet_config.yaml
echo Data directory: %DATA_DIR%
echo Number of GPUs: %NUM_GPUS%
echo.

python train_detectron.py --config configs/vitdet_config.yaml --num-gpus %NUM_GPUS%

if errorlevel 1 (
    echo Training failed!
    pause
    exit /b 1
)

echo Training completed successfully!
echo Check logs_vitdet directory for results
echo Monitor training with: tensorboard --logdir logs_vitdet
pause
exit /b 0

:data_error
echo.
echo Required data structure:
echo %DATA_DIR%\
echo   ^|-- train.csv
echo   ^|-- train_images\
echo   ^|   ^|-- image_001.tiff
echo   ^|   ^`-- ...
echo   ^`-- train_label_masks\
echo       ^|-- image_001_mask.tiff
echo       ^`-- ...
pause
exit /b 1
