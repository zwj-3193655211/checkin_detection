@echo off
cd /d "%~dp0"
python scripts\feature_label_tool.py --picture_dir data\raw
pause
