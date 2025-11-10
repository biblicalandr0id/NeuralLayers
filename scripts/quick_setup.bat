@echo off
REM NeuralLayers Quick Setup Script for Windows

echo ==========================================================================
echo 🚀 NeuralLayers Quick Setup (Windows)
echo ==========================================================================

REM Check Python installation
echo.
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo ✅ Python found

REM Check if virtual environment exists
echo.
echo Checking for virtual environment...
if exist venv\ (
    echo ✅ Virtual environment found
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    echo Please activate manually: venv\Scripts\activate.bat
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip -q
echo ✅ pip upgraded

REM Install dependencies
echo.
echo Installing dependencies...
if exist requirements.txt (
    pip install -r requirements.txt -q
    echo ✅ Dependencies installed
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Install NeuralLayers in development mode
echo.
echo Installing NeuralLayers in development mode...
pip install -e . -q
echo ✅ NeuralLayers installed

REM Run health check
echo.
echo ==========================================================================
echo 🏥 Running Health Check
echo ==========================================================================
python health_check.py

REM Print next steps
echo.
echo ==========================================================================
echo ✨ Setup Complete!
echo ==========================================================================
echo.
echo Next steps:
echo.
echo   1. Activate the environment:
echo      venv\Scripts\activate.bat
echo.
echo   2. Try the examples:
echo      python examples\simple_network.py
echo      python examples\basic_training.py
echo.
echo   3. Run benchmarks:
echo      cd benchmarks ^&^& python benchmark_inference.py
echo.
echo   4. Start Jupyter:
echo      jupyter lab notebooks\
echo.
echo   5. Run tests:
echo      pytest tests\ -v
echo.
echo   6. Launch demo:
echo      python demo_app.py --mode streamlit
echo.
echo ==========================================================================
echo 📚 Documentation: README.md
echo 🐛 Issues: https://github.com/biblicalandr0id/NeuralLayers/issues
echo ==========================================================================

pause
