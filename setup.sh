#!/bin/bash
# 晨读晨练签到检测系统 - 环境安装脚本 (Linux/Mac)

echo "================================================"
echo "晨读晨练签到检测系统 - 环境安装"
echo "================================================"
echo ""

echo "[1/5] 创建conda环境: checkin_detection..."
conda create -n checkin_detection python=3.10 -y

echo ""
echo "[2/5] 激活环境并安装PyTorch..."
source ~/.bashrc  # 确保conda可用
conda activate checkin_detection

echo ""
echo "[3/5] 安装基础依赖..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[4/5] 安装CLIP (用于特征提取)..."
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "[5/5] 安装项目依赖..."
pip install pillow pandas numpy scikit-learn

echo ""
echo "创建数据目录结构..."
mkdir -p data/raw
mkdir -p outputs

echo ""
echo "================================================"
echo "环境创建完成！"
echo "================================================"
echo ""
echo "激活环境: conda activate checkin_detection"
echo ""
echo "快速开始:"
echo "  1. 运行 python src/checkin_system.py 启动主程序"
echo "  2. 或运行 python test_system.py 测试系统"
echo "  3. 如需重新训练，运行 python train_mlp.py"
echo ""
