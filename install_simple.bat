@echo off
title Fire Detection System Installer
color 0A

echo ========================================
echo    FIRE DETECTION SYSTEM INSTALLER
echo ========================================
echo.

echo [1/2] Installing OpenCV and NumPy...
pip install opencv-python numpy

echo.
echo [2/2] Installation Complete!
echo.
echo Starting Fire Detection System...
echo.

python main_simple.py

pause