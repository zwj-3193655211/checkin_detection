@echo off
cd /d "%~dp0"
python scripts\label_tool.py --picture_dir data\raw
pause