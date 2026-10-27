#!/bin/bash
# 启动 Gradio 界面的便捷脚本

echo "激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

echo "启动 Gradio 界面..."
python app_gradio.py
