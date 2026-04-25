@echo off
REM 创建晨读晨练签到检测系统Conda环境

echo 创建conda环境: checkin_detection...
conda create -n checkin_detection python=3.10 -y

echo.
echo 激活环境并安装PyTorch...
conda activate checkin_detection

echo.
echo 安装基础依赖...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 安装项目依赖...
pip install -r requirements.txt

echo.
echo 安装CLIP (用于特征提取)...
pip install git+https://github.com/openai/CLIP.git

echo.
echo 创建数据目录结构...
if not exist "data\raw" mkdir "data\raw"
if not exist "outputs" mkdir "outputs"

echo.
echo ================================================
echo 环境创建完成！
echo ================================================
echo.
echo 激活环境: conda activate checkin_detection
echo.
echo 下一步:
echo 1. 将图片放入 data\raw 目录
echo 2. 运行 python scripts\preprocessing.py 进行预处理
echo 3. 运行 python src\train_resnet.py 训练模型
echo 4. 运行 python src\checkin_system.py 启动主程序
echo.

pause
