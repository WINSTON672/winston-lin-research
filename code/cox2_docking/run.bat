@echo off
REM COX-2 docking pipeline — Strix launcher
REM Double-click this file to run.

cd /d "%~dp0"

echo === Installing Python deps (one-time) ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo === Checking for Uni-Dock (CUDA) ===
where unidock >nul 2>&1
if %errorlevel%==0 (
    echo Uni-Dock found - will use GPU.
) else (
    echo Uni-Dock NOT found - falling back to CPU Vina.
    echo To install Uni-Dock for ~30x speedup on RTX 5080:
    echo   1. conda install -c conda-forge unidock
    echo      (or download binary: https://github.com/dptech-corp/Uni-Dock/releases^)
    echo   2. Make sure 'unidock.exe' is in PATH
)

echo.
echo === Running docking pipeline ===
python dock_pipeline.py

echo.
echo === Done ===
pause
