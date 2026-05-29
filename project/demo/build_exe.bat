@echo off
setlocal
cd /d "%~dp0\.."
pyinstaller --onefile --windowed --name FacialSegmentationDemo demo\face_segmentation_demo.py
endlocal
