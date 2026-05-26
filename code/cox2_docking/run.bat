@echo off
REM COX-2 docking pipeline — Strix launcher
REM Double-click this file to run.

cd /d "%~dp0"

echo === Installing Python deps (one-time) ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo === Checking for docking engine ===
where unidock >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Uni-Dock found - will use GPU.
    goto run
)
where vina >nul 2>&1
if %errorlevel%==0 (
    echo [OK] vina.exe found - will use CPU.
    goto run
)
echo.
echo [X] NO DOCKING ENGINE FOUND
echo.
echo Install ONE of these:
echo.
echo   GPU path (fast, recommended for RTX 5080):
echo     1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
echo     2. Open Anaconda Prompt
echo     3. conda install -c conda-forge unidock
echo.
echo   CPU path (slower, simpler):
echo     1. Download vina.exe from:
echo        https://github.com/ccsb-scripps/AutoDock-Vina/releases
echo     2. Put vina.exe in C:\Windows\System32
echo        (or anywhere in PATH)
echo.
pause
exit /b 1

:run
echo.
echo === Running docking pipeline ===
python dock_pipeline.py

echo.
echo === Done ===
pause
