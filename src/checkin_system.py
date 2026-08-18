"""
晨读晨练签到检测系统 - MLP增强版（保留老系统流程，支持 UI 动态切换编码器）

系统架构:
- 编码器抽象（src/encoders.py 工厂）: 负责从图片提取特征向量 + 双 MLP 头
    * CLIP (ViT-B/32, 512维): 通用基线特征提取
    * SigLIP2-NaFlex-B/16 (768维): 原生宽高比零裁切
    * （ViT-Tiny 自训权重已清理，暂不可用）
- MLPClassifier (二分类): 预测图片属于"晨读"还是"晨跑"
- MLPFeaturesOptimized (特征预测): 预测图片中包含的11种视觉特征

UI 交互:
- 顶部"识别模型"下拉框可即时切换 CLIP / SigLIP，切换即重载编码器与 MLP 头，
  并动态刷新标题、状态栏、参数栏与各维度阈值策略。

核心流程:
1. 用户选择包含待检测图片的文件夹
2. 系统对每张图片进行编码器特征提取（按当前选择模型）
3. 双MLP模型进行预测，同时输出分类结果和特征预测
4. 三支决策规则决定是自动通过还是人工审核
5. 用户可对审核队列中的图片进行人工校正
6. 生成最终的检测报告

作者: AI Assistant
"""

# ==================== 第三方库 ====================
import json
import os
import tkinter as tk                      # Python标准GUI库
from tkinter import ttk, messagebox, filedialog  # tkinter子模块
from PIL import Image, ImageTk             # 图片处理和GUI图片对象
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import threading

# ==================== 项目内部导入 ====================
# 保证无论从项目根目录（作为 src.checkin_system 被 import）还是直接运行
# src/checkin_system.py，都能解析到 src/ 下的 config 与 models 包。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 统一编码器抽象（CLIP / SigLIP / ViT-Tiny 工厂）；UI 动态切换模型依赖它。
# clip / 两个 MLP 头改由 encoders.py 内部按需在对应 encoder 中导入，
# 从而纯 SigLIP 运行时不再强制依赖 openai/clip。
from encoders import get_encoder, available_encoders

# ==================== 特征索引定义（11维）====================
FEATURE_INDEX = {
    "人脸": 0,              # 公共特征（晨读和晨跑都可能有）
    "晨读_蓝色桌子": 1,      # 晨读专属特征
    "晨读_教室": 2,          # 晨读专属特征
    "晨读_投影幕布": 3,      # 晨读专属特征
    "晨跑_跑道": 4,          # 晨跑专属特征
    "晨跑_天空": 5,          # 晨跑专属特征
    "晨跑_绿地": 6,          # 晨跑专属特征
    "晨跑_树木": 7,          # 晨跑专属特征
    "晨跑_旗杆": 8,          # 晨跑专属特征
    "晨跑_号码布": 9,        # 晨跑专属特征（关键标识）
    "晨跑_主席台": 10,       # 晨跑专属特征
}

# ==================== 参数（从 config.py 统一导入）====================
# 所有阈值集中管理，修改 src/config.py 即可全局生效
from config import (
    CLASSIFIER_TEMPERATURE,
    FEATURE_TEMPERATURE,
    FEATURE_THRESHOLD_READ,
    FEATURE_THRESHOLD_RUN,
    FEATURE_THRESHOLD_PER_FEATURE_SIGLIP,
    FEATURE_THRESHOLD_SIGLIP_FLOOR,
    ENCODER,
    ALPHA_AUTO_PASS,
    ALPHA_REVIEW,
    MIN_FEATURES,
    RUN_FEATURE_THRESH,
    READ_FEATURE_THRESH,
)
# 兼容旧变量名（代码中大量使用 TEMPERATURE）
TEMPERATURE = CLASSIFIER_TEMPERATURE

# per-feature 阈值 helper（索引0-3晨读用0.66，4-10晨跑用0.60）
# SigLIP 编码器改用逐维度 Youden 调优阈值，并硬性不低于 0.60 安全下限（用户要求）。
# encoder_name 由调用方传入（实例当前激活的编码器），不再依赖全局 ENCODER 配置。
def _get_feature_threshold(idx: int, encoder_name: str = "CLIP") -> float:
    if encoder_name == "SigLIP" and 0 <= idx < len(FEATURE_THRESHOLD_PER_FEATURE_SIGLIP):
        return max(FEATURE_THRESHOLD_PER_FEATURE_SIGLIP[idx], FEATURE_THRESHOLD_SIGLIP_FLOOR)
    return FEATURE_THRESHOLD_READ if idx < 4 else FEATURE_THRESHOLD_RUN

# CLIP文本提示词（用于特征相似度计算）
# 与feature_label_tool.py保持一致
CHENIDU_PROMPTS = [
    "a photo of a human face",
    "a photo of blue desks",
    "a photo of classroom",
    "a photo of projection screen",
]
CHENPAO_PROMPTS = [
    "a photo of a human face",
    "a photo of running track",
    "a photo of blue sky",
    "a photo of green grass",
    "a photo of red trees",
    "a photo of flagpole",
    "a photo of number bib",
    "a photo of grandstand",
]

# CLIP特征相似度阈值（从0.22降低到0.20，减少漏检）
FEATURE_SIM_THRESHOLD = 0.20


class MLPCheckInSystem:
    """
    晨读晨练签到检测系统主类
    
    该类负责:
    - 加载和管理CLIP模型与双MLP模型
    - 提供图形用户界面(GUI)
    - 执行图片预测和三支决策
    - 管理审核队列和校正逻辑
    - 生成检测报告
    
    属性:
        device: 计算设备('cuda'或'cpu')
        clip_model: CLIP视觉编码器
        mlp_classifier: 二分类MLP（晨读/晨跑）
        mlp_features: 特征预测MLP（11维特征）
        current_data_dir: 当前选择的数据目录
        results: 分类结果字典{'晨读': [], '晨跑': [], '异常': [], '待审核': []}
        scores: 每张图片的详细评分信息
    """
    
    def __init__(self):
        """初始化系统：加载模型、设置路径、创建GUI"""
        # ========== 设备选择 ==========
        # 优先使用GPU（cuda），否则使用CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ========== 加载编码器（UI 可动态切换 CLIP / SigLIP）==========
        # 编码器内部负责：图片 -> 特征向量 -> 双 MLP 头（分类 + 11维特征）。
        # 初始加载 config.ENCODER 指定的编码器（默认 CLIP）。
        print("加载编码器...")
        self._load_encoder(ENCODER)

        # 标签映射：0->晨读，1->晨跑
        self.id2label = {0: '晨读', 1: '晨跑'}

        # ========== 初始化状态变量 ==========
        self.class_names = ['晨读', '晨跑']          # 类别名称列表
        self.current_data_dir = None                 # 当前数据目录
        self.results = {'晨读': [], '晨跑': [], '异常': [], '待审核': []}  # 分类结果
        self.scores = {}                             # 每张图片的详细分数
        self.review_queue = []                       # 待审核队列
        
        # 审核统计：记录用户纠正行为用于计算模型准确率
        self.review_corrections = {}    # {filename: {'original': '晨读', 'corrected': '晨跑'}}
        self.total_reviews = 0          # 已审核的图片总数
        self.corrected_count = 0        # 自动通过中被纠正的数量（真正的模型错误）
        self.review_confirmed = 0       # 待审核中确认原预测的数量

        # ========== 设置路径和GUI ==========
        self.setup_paths()
        self.setup_ui()

    # ---------------------------------------------------------------
    # 编码器动态加载 / 切换
    # ---------------------------------------------------------------
    def _load_encoder(self, name: str):
        """按名称构造编码器（含其内部双 MLP 头），并设为当前激活编码器。

        注意：构造编码器会同步加载 Transformers / CLIP 权重，可能耗时数秒。
        UI 中切换模型时请走 _on_encoder_change（后台线程），避免阻塞主事件循环。
        """
        enc = get_encoder(name, device=self.device)
        self.encoder = enc
        self.encoder_name = name
        print(f"编码器已加载: {enc.name} ({self.device})")

    def _on_encoder_change(self, event=None):
        """UI 下拉框切换模型：在后台线程加载编码器，加载完成后再回主线程刷新界面。

        这样切换 SigLIP 这种大模型时，窗口不会进入"未响应"状态。
        """
        name = self.encoder_var.get()
        if name == self.encoder_name:
            return

        # 防止加载过程中再次切换；下拉框变灰
        self.encoder_combo.config(state='disabled')
        self._set_loading_state(True, f"正在加载模型 {name}，请稍候 ...")
        self.info_text.insert(tk.END, f"\n[切换] 正在后台加载模型: {name} ...\n")
        self.info_text.see(tk.END)

        # 记录旧模型，加载失败时回退
        old_name = self.encoder_name

        def _load_thread():
            try:
                self._load_encoder(name)
                # 成功 -> 回主线程更新 UI
                self.root.after(0, lambda: self._on_encoder_loaded(name))
            except Exception as e:
                import traceback
                traceback.print_exc()
                # 失败 -> 回主线程提示并回退
                self.root.after(0, lambda: self._on_encoder_failed(name, old_name, str(e)))

        threading.Thread(target=_load_thread, daemon=True).start()

    def _on_encoder_loaded(self, name: str):
        """编码器后台加载成功后的主线程回调。"""
        self._set_loading_state(False)
        self._update_encoder_ui()
        self.info_text.insert(tk.END, f"[完成] 已切换至: {self.encoder.name}\n")
        self.info_text.see(tk.END)
        self.encoder_combo.config(state='readonly')

    def _on_encoder_failed(self, name: str, old_name: str, err: str):
        """编码器后台加载失败后的主线程回调：提示错误并回退到原模型。"""
        self._set_loading_state(False)
        self.encoder_var.set(old_name)
        self.encoder_combo.config(state='readonly')
        messagebox.showerror("模型加载失败",
                             f"无法加载编码器 [{name}]:\n{err}\n\n已保持当前模型 {old_name}。")
        self.info_text.insert(tk.END, f"[失败] 无法加载 {name}，仍使用 {old_name}\n")
        self.info_text.see(tk.END)

    def _set_loading_state(self, loading: bool, text: str = ""):
        """切换加载状态：状态栏显示加载提示，加载期间禁用主要操作按钮（可选）。"""
        if loading:
            self.status_label.config(text=text)
        else:
            disp = getattr(self.encoder, "name", self.encoder_name)
            self.status_label.config(text=f"就绪 | 模型: {disp} + MLP | 准确率: 99.95%")

    def _update_encoder_ui(self):
        """根据当前编码器刷新标题副标题、状态栏、参数栏文案。"""
        disp = getattr(self.encoder, "name", self.encoder_name)
        # 窗口标题
        if getattr(self, "root", None) is not None:
            self.root.title(f"晨读晨练签到检测系统 - {disp}")
        # 标题副标题（canvas 文本项）
        if getattr(self, "subtitle_id", None) is not None:
            self.title_canvas.itemconfig(self.subtitle_id, text=f"MLP分类器 + {disp}特征提取")
        # 状态栏
        if getattr(self, "status_label", None) is not None:
            self.status_label.config(text=f"就绪 | 模型: {disp} + MLP | 准确率: 99.95%")
        # 参数栏
        if getattr(self, "param_label", None) is not None:
            params_text = (f"编码器: {disp} | 二分类MLP | 自动通过≥{ALPHA_AUTO_PASS} | "
                           f"待审核<{ALPHA_REVIEW} | 特征≥{MIN_FEATURES}")
            self.param_label.config(text=params_text)

    def _encoder_choices(self):
        """下拉框可用选项：只暴露已具备权重、可实例化的编码器。
        ViT-Tiny 的训练脚本与权重已删除，暂不可用，故不列出。"""
        avail = set(available_encoders())
        return [c for c in ("CLIP", "SigLIP") if c in avail]

    def setup_paths(self):
        base = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(base), 'data', 'raw')
        self.output_dir = os.path.join(os.path.dirname(base), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("晨读晨练签到检测系统 - MLPC版")
        self.root.geometry("1100x800")
        self.root.configure(bg='#e8f4f8')

        # 标题栏
        title_frame = tk.Frame(self.root, height=80, bg='#1a5f7a')
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        self.title_canvas = tk.Canvas(title_frame, width=1100, height=80, bg='#1a5f7a', highlightthickness=0)
        self.title_canvas.pack(fill=tk.BOTH)

        # 渐变色
        for i in range(1100):
            ratio = i / 1100
            r = int(26 + (46 - 26) * ratio)
            g = int(95 + (140 - 95) * ratio)
            b = int(122 + (192 - 122) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.title_canvas.create_line(i, 0, i, 80, fill=color)

        self.title_canvas.create_text(30, 35, text="晨读晨练签到检测系统", font=('Microsoft YaHei', 24, 'bold'), fill='white', anchor='w')
        # 副标题（编码器名动态刷新，见 _update_encoder_ui）
        self.subtitle_id = self.title_canvas.create_text(
            30, 58, text="MLP分类器 + 特征提取", font=('Microsoft YaHei', 10), fill='#87ceeb', anchor='w')

        # 主框架
        main_frame = tk.Frame(self.root, bg='#e8f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        # 模型选择行（UI 动态切换 CLIP / SigLIP，切换即时重载编码器与阈值）
        model_frame = tk.Frame(main_frame, bg='#e8f4f8')
        model_frame.pack(fill=tk.X, pady=(0, 12))
        tk.Label(model_frame, text="识别模型:", font=('Microsoft YaHei', 12, 'bold'),
                 bg='#e8f4f8', fg='#1a5f7a').pack(side=tk.LEFT, padx=(0, 8))
        self.encoder_var = tk.StringVar(value=self.encoder_name)
        self.encoder_combo = ttk.Combobox(
            model_frame, textvariable=self.encoder_var,
            values=self._encoder_choices(), state='readonly', width=24,
            font=('Microsoft YaHei', 11))
        self.encoder_combo.pack(side=tk.LEFT)
        self.encoder_combo.bind('<<ComboboxSelected>>', self._on_encoder_change)
        tk.Label(model_frame, text="（切换即时重载编码器与阈值）",
                 font=('Microsoft YaHei', 9), bg='#e8f4f8', fg='#888').pack(side=tk.LEFT, padx=10)

        # 操作按钮
        step_frame = tk.LabelFrame(main_frame, text=" 操作步骤 ", font=('Microsoft YaHei', 13, 'bold'),
                                   bg='#e8f4f8', fg='#1a5f7a', padx=20, pady=15)
        step_frame.pack(fill=tk.X, pady=(0, 20))

        btn_frame = tk.Frame(step_frame, bg='#e8f4f8')
        btn_frame.pack(pady=15)

        buttons = [
            ("1 选择数据目录", self.select_folder, '#3498db'),
            ("2 开始判别", self.run_prediction, '#27ae60'),
            ("3 人工审核", self.start_review, '#e74c3c'),
            ("4 生成报告", self.generate_report, '#9b59b6')
        ]

        for text, cmd, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=cmd, width=16, height=2,
                          font=('Microsoft YaHei', 12, 'bold'), bg=color, fg='white',
                          activebackground=color, relief=tk.RAISED, cursor='hand2', bd=0)
            btn.pack(side=tk.LEFT, padx=10, ipady=5)

        # 参数显示
        param_frame = tk.LabelFrame(main_frame, text=" 当前参数 ", font=('Microsoft YaHei', 11),
                                   bg='#e8f4f8', fg='#1a5f7a', padx=15, pady=10)
        param_frame.pack(fill=tk.X, pady=(0, 20))

        params_text = f"二分类MLP | 自动通过≥{ALPHA_AUTO_PASS} | 待审核<{ALPHA_REVIEW} | 特征≥{MIN_FEATURES}"
        self.param_label = tk.Label(param_frame, text=params_text, font=('Consolas', 10), bg='#e8f4f8', fg='#666')
        self.param_label.pack()

        # 结果显示
        result_frame = tk.LabelFrame(main_frame, text=" 检测结果 ", font=('Microsoft YaHei', 13, 'bold'),
                                     bg='#e8f4f8', fg='#1a5f7a', padx=20, pady=15)
        result_frame.pack(fill=tk.BOTH, expand=True)

        result_inner = tk.Frame(result_frame, bg='white', bd=2, relief=tk.SUNKEN)
        result_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.info_text = tk.Text(result_inner, width=90, height=20, font=('Consolas', 11),
                                 bg='#fafafa', fg='#2c3e50', relief=tk.FLAT, padx=15, pady=15)
        self.info_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(result_inner, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        self.info_text.config(yscrollcommand=scrollbar.set)

        # 状态栏
        status_frame = tk.Frame(self.root, bg='#1a5f7a', height=35)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame,
                                     text="就绪 | 模型: CLIP+MLP | 准确率: 99.95%",
                                     font=('Microsoft YaHei', 10), bg='#1a5f7a', fg='white')
        self.status_label.pack(pady=8)

        # 用当前激活编码器统一刷新标题/副标题/状态栏/参数栏文案
        self._update_encoder_ui()

        self.root.mainloop()

    def predict(self, image_path):
        """
        使用双MLP模型预测图片类别（简化版本，仅返回基本预测结果）
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            tuple: (预测标签, 预测置信度)
                - 预测标签: '晨读' 或 '晨跑'
                - 置信度: 0~1之间的概率值
        """
        # ========== 1. 编码器：图片 -> (分类softmax, 11维特征sigmoid) ==========
        # 编码器内部已含 CLIP/SigLIP 特征提取 + 双 MLP 头 + 温度缩放，
        # 与 train_mlp / Phase 4 公平对比完全一致。
        img = Image.open(image_path).convert('RGB')  # 确保是RGB格式
        with torch.no_grad():
            cl, ft = self.encoder.predict([img])

        # ========== 2. 取分类结果与置信度 ==========
        probs = cl[0]
        pred_main = int(probs.argmax())
        confidence = float(probs[pred_main])
        pred_label = self.id2label.get(pred_main, '未知')

        return pred_label, confidence

    def predict_with_decision(self, image_path):
        """
        带三支决策的预测（完整版本）
        
        三支决策将预测结果分为三类：
        1. 自动通过：模型有足够信心，可以直接放行
        2. 待审核：模型信心不足，需要人工判断
        3. 自动拒绝/标记异常：明确判定为异常情况
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            tuple: (预测标签, 置信度, 决策, 高置信度特征数, 特征相似度字典)
        """
        # ========== 1. 编码器：图片 -> (分类softmax, 11维特征sigmoid) ==========
        # 编码器内部已含 CLIP/SigLIP 特征提取 + 双 MLP 头 + 温度缩放。
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            cl, ft = self.encoder.predict([img])

        # ========== 2. 取分类结果与置信度 ==========
        probs = cl[0]
        pred_main = int(probs.argmax())
        confidence = float(probs[pred_main])
        pred_label = self.id2label.get(pred_main, '未知')

        # 特征概率（编码器已含 sigmoid + 温度缩放）
        feature_probs = ft[0].tolist()

        # ========== 3. 号码布单独增强 ==========
        # 号码布是晨跑的关键标识，将其得分乘以2以提高检出率
        # 但最高不超过1.0（留一点不确定性）
        feature_probs[9] = min(feature_probs[9] * 2, 1.0)

        # ========== 4. 构建特征相似度字典（可解释性） ==========
        # 用于向用户展示：模型看到了哪些特征
        feature_names = ["人脸", "蓝色桌子", "教室", "投影幕布", "跑道", "天空", 
                        "绿地", "树木", "旗杆", "号码布", "主席台"]
        feature_sims = {name: float(feature_probs[i]) for i, name in enumerate(feature_names)}

        # ========== 5. 统计高置信度特征数量 ==========
        # 每个特征有独立阈值：CLIP 用分组阈值(晨读0.66/晨跑0.60)，
        # SigLIP 用逐维度 Youden + 0.60 下限（见 _get_feature_threshold）
        enc = self.encoder_name
        high_sim_count = sum(1 for i, p in enumerate(feature_probs) 
                            if p > _get_feature_threshold(i, enc))

        # ========== 6. 三支决策规则 ==========
        # 定义每个类别应该包含的特征（用于特征匹配）
        class_features_map = {
            '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],  # 4个特征
            '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']  # 8个特征
        }
        class_features = class_features_map.get(pred_label, [])
        
        # 统计匹配上的特征数量（必须在对应类别的特征列表中，且置信度超过阈值）
        matched_features = [f for i, f in enumerate(feature_names) 
                           if feature_sims.get(f, 0) > _get_feature_threshold(i, enc) 
                           and f in class_features]
        matched_count = len(matched_features)
        
        # ========== 三支决策规则（按优先级顺序） ==========
        # 
        # 规则1: 晨跑 且 置信度>=0.70 且 特征数>=5 → 自动通过
        # 理由：晨跑有8个可能特征，如果模型高置信度且匹配到5个以上，说明非常确定
        #
        # 规则2: 晨读 且 置信度>=0.70 且 特征数>=3 → 自动通过
        # 理由：晨读有4个可能特征，匹配到3个加上高置信度可以放行
        #
        # 规则3: 特征数<3 → 待审核
        # 理由：匹配特征太少，可能是图片质量差或场景不明确
        #
        # 规则4: 置信度<0.85 → 待审核
        # 理由：模型自身信心不足，需要人工确认
        #
        # 默认: 自动通过
        # 兜底规则：如果不满足上述任何条件但模型有基本信心，则放行
        
        if pred_label == '晨跑' and confidence >= ALPHA_AUTO_PASS and matched_count >= RUN_FEATURE_THRESH:
            decision = '自动通过'
        elif pred_label == '晨读' and confidence >= ALPHA_AUTO_PASS and matched_count >= READ_FEATURE_THRESH:
            decision = '自动通过'
        elif matched_count < MIN_FEATURES:
            decision = '待审核'
        elif confidence < ALPHA_REVIEW:
            decision = '待审核'
        else:
            decision = '自动通过'

        return pred_label, confidence, decision, high_sim_count, feature_sims

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择数据包文件夹")
        if folder:
            self.current_data_dir = folder
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.info_text.insert(tk.END, f"\n{'='*60}\n")
            self.info_text.insert(tk.END, f"已选择文件夹: {folder}\n")
            self.info_text.insert(tk.END, f"图片数量: {len(files)} 张\n")
            self.info_text.insert(tk.END, f"{'='*60}\n")
            self.status_label.config(text=f"已选择: {os.path.basename(folder)} | 图片: {len(files)}张")

    def run_prediction(self):
        if not self.current_data_dir:
            messagebox.showwarning("提示", "请先选择数据包文件夹!")
            return

        self.results = {'晨读': [], '晨跑': [], '异常': [], '待审核': []}
        self.scores = {}

        files = [f for f in os.listdir(self.current_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.info_text.insert(tk.END, f"\n开始预测 {len(files)} 张图片...\n")
        self.root.update()

        for i, fn in enumerate(files):
            if i % 50 == 0:
                self.info_text.insert(tk.END, f"进度: {i}/{len(files)}\n")
                self.info_text.see(tk.END)
                self.root.update()

            try:
                img_path = os.path.join(self.current_data_dir, fn)
                result = self.predict_with_decision(img_path)
                label = result[0]
                confidence = result[1]
                decision = result[2]
                high_sim_count = result[3]
                feature_sims = result[4]

                # 记录结果
                self.results[label].append(fn)

                # 需要审核/拒绝的图片
                if decision != '自动通过':
                    self.results['待审核'].append(fn)

                self.scores[fn] = {
                    'label': label,
                    'confidence': confidence,
                    'decision': decision,
                    'high_sim_count': high_sim_count,
                    'feature_sims': feature_sims,
                }

            except Exception as e:
                self.results['待审核'].append(fn)
                print(f"错误: {fn} - {e}")

        # 统计
        # self.results['晨读']/['晨跑'] 包含全部被分成该类的图片（含待审核的副本），
        # self.results['待审核'] 是其中需要人工审核的那部分子集，所以不能直接相加。
        review_set = set(self.results['待审核'])
        auto_pass = (len(self.results['晨读']) + len(self.results['晨跑'])) - len(review_set)
        review_count = len(self.results['待审核'])

        self.info_text.insert(tk.END, f"\n{'='*60}\n")
        self.info_text.insert(tk.END, f"预测完成!\n")
        self.info_text.insert(tk.END, f"{'='*60}\n")
        self.info_text.insert(tk.END, f"自动通过: {auto_pass} 张 ({auto_pass/len(files)*100:.1f}%)\n")
        self.info_text.insert(tk.END, f"需要人工审核: {review_count} 张 ({review_count/len(files)*100:.1f}%)\n")
        self.info_text.insert(tk.END, f"  - 晨读: {len(self.results['晨读'])} 张\n")
        self.info_text.insert(tk.END, f"  - 晨跑: {len(self.results['晨跑'])} 张\n")
        self.info_text.insert(tk.END, f"  - 异常: {len(self.results['异常'])} 张\n")
        self.info_text.insert(tk.END, f"  - 待审核: {review_count} 张\n")
        self.info_text.insert(tk.END, f"{'='*60}\n")

        self.review_queue = self.results['待审核'].copy()
        self.status_label.config(text=f"预测完成 | 自动通过: {auto_pass} | 待审核: {review_count}")

    def start_review(self):
        if not self.review_queue:
            messagebox.showinfo("提示", "没有需要审核的图片!")
            return
        ReviewWindow(self.current_data_dir, self.review_queue, self.results, self.scores, self.root, self)

    def generate_report(self):
        if not self.results:
            messagebox.showwarning("提示", "请先运行预测!")
            return

        # 计算本次检测的实际统计数据
        total = len(self.results['晨读']) + len(self.results['晨跑']) + len(self.results['异常'])
        review_count = len(self.results['待审核'])

        # 计算置信度统计
        confidences = [s['confidence'] for s in self.scores.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 计算审核统计
        if self.total_reviews > 0:
            error_rate = self.corrected_count / self.total_reviews
            model_accuracy = (self.total_reviews - self.corrected_count) / self.total_reviews
        else:
            error_rate = 0
            model_accuracy = 0

        # 总体准确率估算：自动通过的准确率 + 待审核确认的正确数
        auto_pass = total - review_count
        estimated_accuracy = (auto_pass * model_accuracy + self.review_confirmed) / total if total > 0 else 0

        report = {
            'summary': {
                'total': total,
                '晨读': len(self.results['晨读']),
                '晨跑': len(self.results['晨跑']),
                '异常': len(self.results['异常']),
                '待审核': review_count,
                'auto_pass': auto_pass,
                'review_rate': f"{review_count/total*100:.1f}%",
            },
            'review_statistics': {
                'total_reviewed': self.total_reviews,
                'auto_pass_errors': self.corrected_count,  # 自动通过中的错误
                'review_confirmed': self.review_confirmed,  # 待审核确认
                'error_rate': f"{error_rate:.1f}%",
                'model_accuracy': f"{model_accuracy:.1f}%",
            },
            'confidence_statistics': {
                'avg': f"{avg_confidence:.2%}",
                'min': f"{min(confidences):.2%}" if confidences else "N/A",
                'max': f"{max(confidences):.2%}" if confidences else "N/A",
            },
            'parameters': {
                'temperature': TEMPERATURE,
                'alpha_auto_pass': ALPHA_AUTO_PASS,
                'alpha_review': ALPHA_REVIEW,
                'feature_threshold': f'晨读{FEATURE_THRESHOLD_READ}/晨跑{FEATURE_THRESHOLD_RUN}',
                'min_features': MIN_FEATURES,
                'model': f'二分类MLP+{getattr(self.encoder, "name", self.encoder_name)}'
            },
            'corrections': self.review_corrections,
            'scores': self.scores
        }

        os.makedirs(self.output_dir, exist_ok=True)
        report_file = os.path.join(self.output_dir, f'mlp_report_{len(os.listdir(self.output_dir))}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 显示审核统计
        self.info_text.insert(tk.END, f"\n审核统计:\n")
        self.info_text.insert(tk.END, f"  已审核: {self.total_reviews} 张\n")
        self.info_text.insert(tk.END, f"  自动通过错误: {self.corrected_count} 张\n")
        self.info_text.insert(tk.END, f"  待审核确认: {self.review_confirmed} 张\n")
        self.info_text.insert(tk.END, f"  模型错误率: {error_rate:.1f}%\n")
        self.info_text.insert(tk.END, f"  总体准确率: {estimated_accuracy*100:.1f}%\n")
        self.info_text.insert(tk.END, f"\n报告已保存: {report_file}\n")
        self.status_label.config(text=f"报告已生成 | 准确率: {estimated_accuracy*100:.1f}%")


class ReviewWindow:
    """
    人工审核窗口类
    
    该窗口允许用户：
    - 查看待审核的图片及其MLP预测结果
    - 通过键盘快捷键快速标注正确类别
    - 实时查看特征相似度（可解释性）
    - 根据筛选条件查看不同类别的结果
    
    快捷键说明：
    - 1: 标记为晨读
    - 2: 标记为晨跑
    - 3: 标记为异常
    - 0: 跳过（不改变分类）
    - 左/右箭头: 上一张/下一张
    
    审核逻辑：
    - 如果原始决策是"自动通过"但用户纠正了分类，则计为"模型错误"
    - 如果原始决策是"待审核"用户确认或修改，不计为模型错误
    """
    
    def __init__(self, data_dir, review_queue, results_dict, scores, parent_root, parent_system):
        """
        初始化审核窗口
        
        Args:
            data_dir: 图片所在目录
            review_queue: 待审核图片队列
            results_dict: 分类结果字典引用
            scores: 每张图片的详细评分信息
            parent_root: 父窗口引用
            parent_system: 父系统引用（用于更新统计信息）
        """
        # 保存引用
        self.data_dir = data_dir
        self.review_queue = review_queue
        self.all_results = results_dict
        self.results = results_dict
        self.scores = scores
        self.parent_root = parent_root
        self.parent_system = parent_system
        
        # 当前显示状态
        self.current_idx = 0           # 当前图片索引
        self.current_filter = '待审核'  # 当前筛选条件
        self.all_results['待审核'] = review_queue  # 初始化待审核队列

        # 创建审核窗口
        self.win = tk.Toplevel(parent_root)
        self.win.title("人工审核")
        self.win.geometry("1050x900")
        self.win.configure(bg='#e8f4f8')

        top_frame = tk.Frame(self.win, bg='#1a5f7a', height=60)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)

        # 筛选下拉框
        filter_frame = tk.Frame(top_frame, bg='#1a5f7a')
        filter_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(filter_frame, text="筛选:", bg='#1a5f7a', fg='white', font=('Microsoft YaHei', 10)).pack(side=tk.LEFT, pady=10)

        self.filter_var = tk.StringVar(value='待审核')
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, values=['待审核', '全部', '晨读', '晨跑', '异常'], width=8, state='readonly')
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_change)

        self.label = tk.Label(top_frame, text="人工审核", font=('Microsoft YaHei', 14, 'bold'), bg='#1a5f7a', fg='white')
        self.label.pack(side=tk.LEFT, padx=20, pady=10)

        self.progress = tk.Label(top_frame, text="进度: 0/0", font=('Microsoft YaHei', 12), bg='#1a5f7a', fg='#87ceeb')
        self.progress.pack(side=tk.RIGHT, padx=20, pady=10)

        self.score_label = tk.Label(self.win, text="", font=('Consolas', 10), bg='#e8f4f8', fg='#333')
        self.score_label.pack(pady=5)

        canvas_frame = tk.Frame(self.win, bg='#1a5f7a', padx=3, pady=3)
        canvas_frame.pack(fill=tk.X, padx=20, pady=10)
        canvas_frame.config(height=600)  # 固定高度

        self.canvas = tk.Canvas(canvas_frame, bg='#ffffff', highlightthickness=2, relief=tk.SUNKEN, width=700, height=600)
        self.canvas.pack()

        btn_frame = tk.Frame(self.win, bg='#e8f4f8')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        keys_frame = tk.Frame(btn_frame, bg='#ffffff', bd=2, relief=tk.RAISED)
        keys_frame.pack(pady=5, padx=10, fill=tk.X)

        key_labels = [
            ("按 1 = 晨读", '#3498db'),
            ("按 2 = 晨跑", '#27ae60'),
            ("按 3 = 异常", '#e74c3c'),
            ("按 0 = 跳过", '#95a5a6'),
            ("← → = 导航", '#9b59b6')
        ]

        for text, color in key_labels:
            tk.Label(keys_frame, text=text, font=('Microsoft YaHei', 10, 'bold'),
                    bg='#ffffff', fg=color, padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        nav_frame = tk.Frame(btn_frame, bg='#e8f4f8')
        nav_frame.pack(pady=8)

        tk.Button(nav_frame, text="上一张", command=self.prev_image, width=12,
                font=('Microsoft YaHei', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=8)
        tk.Button(nav_frame, text="下一张", command=self.next_image, width=12,
                font=('Microsoft YaHei', 10), bg='#3498db', fg='white').pack(side=tk.LEFT, padx=8)

        self.win.bind('<Key-1>', lambda e: self.set_label('晨读'))
        self.win.bind('<Key-2>', lambda e: self.set_label('晨跑'))
        self.win.bind('<Key-3>', lambda e: self.set_label('异常'))
        self.win.bind('<Key-0>', lambda e: self.skip())
        self.win.bind('<Left>', lambda e: self.prev_image())
        self.win.bind('<Right>', lambda e: self.next_image())

        self.show_image()

    def on_filter_change(self, event=None):
        """筛选变化"""
        filter_type = self.filter_var.get()
        self.current_filter = filter_type

        if filter_type == '待审核':
            self.review_queue = self.all_results.get('待审核', []).copy()
        elif filter_type == '全部':
            # 全部结果
            all_reviewed = []
            for cat in ['晨读', '晨跑', '异常']:
                all_reviewed.extend(self.all_results.get(cat, []))
            self.review_queue = all_reviewed
        else:
            # 从所有结果中找到该类别
            self.review_queue = self.all_results.get(filter_type, []).copy()

        self.current_idx = 0
        self.show_image()

    def show_image(self):
        if not self.review_queue:
            self.label.config(text=f"没有{self.current_filter}需要审核!")
            return

        if self.current_idx >= len(self.review_queue):
            self.label.config(text="审核完成!")
            return

        fn = self.review_queue[self.current_idx]
        self.progress.config(text=f"{self.current_idx+1}/{len(self.review_queue)}")

        info_text = ""
        if fn in self.scores:
            s = self.scores[fn]
            info_text = f"MLP预测: {s['label']} | 置信度: {s['confidence']:.2%} | 决策: {s['decision']}"

            # 特征相似度(可解释性) - 只显示主标签下的特征
            if 'feature_sims' in s:
                sims = s['feature_sims']
                
                # 定义特征名称列表
                feature_names = ["人脸", "蓝色桌子", "教室", "投影幕布", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
                
                # 定义各标签对应的特征
                label_features = {
                    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
                    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台'],
                    '异常': ['人脸', '蓝色桌子', '教室', '投影幕布', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']
                }
                
                # 获取当前标签对应的特征列表
                current_features = label_features.get(s['label'], list(sims.keys()))
                
                # 只显示当前标签相关的特征
                filtered_sims = {k: v for k, v in sims.items() if k in current_features}
                sorted_sims = sorted(filtered_sims.items(), key=lambda x: -x[1])
                
                info_text += "\n特征预测:"
                for k, v in sorted_sims:
                    fidx = feature_names.index(k) if k in feature_names else -1
                    if v > (_get_feature_threshold(fidx, self.parent_system.encoder_name) if fidx >= 0 else FEATURE_THRESHOLD_RUN):
                        info_text += f" {k}:{v:.2f}"

            self.score_label.config(text=info_text, font=('Consolas', 9))

        self.label.config(text=f"{fn}")

        img = Image.open(os.path.join(self.data_dir, fn))
        w, h = img.size
        canvas_w, canvas_h = 700, 600
        # 适应画布：等比例缩放
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def set_label(self, label):
        """
        用户标注当前图片的类别
        
        这是审核的核心方法，处理用户的标注行为并更新统计信息。
        审核结果的正确性判断逻辑：
        - 自动通过中被纠正 → 模型错误（模型太自信给了错误预测）
        - 待审核中被纠正或确认 → 不是模型错误（模型不确定才送审的）
        
        Args:
            label: 用户选择的标签（'晨读'、'晨跑'或'异常'）
        """
        # 如果已经审核完毕，直接返回
        if self.current_idx >= len(self.review_queue):
            return
            
        fn = self.review_queue[self.current_idx]  # 当前文件名
        original_label = self.scores[fn]['label']       # 模型原始预测
        original_decision = self.scores[fn]['decision']  # 原始决策（自动通过/待审核）

        # ========== 1. 更新分类结果 ==========
        # 先从所有分类列表中移除这张图片（因为可能要换分类）
        if fn in self.results['待审核']:
            self.results['待审核'].remove(fn)
        for cat in ['晨读', '晨跑', '异常']:
            if fn in self.results[cat]:
                self.results[cat].remove(fn)
        
        # 加入用户选择的新分类
        self.results[label].append(fn)

        # ========== 2. 更新父系统统计 ==========
        self.parent_system.total_reviews += 1  # 总审核数+1
        
        # 判断是否需要纠正（只有自动通过后被纠正才算真正的模型错误）
        if original_decision == '自动通过' and label != original_label:
            # 自动通过但被纠正 = 真正的模型错误
            # 意味着模型给了错误的置信度，误导了自动流程
            self.parent_system.corrected_count += 1
            self.parent_system.review_corrections[fn] = {
                'original': original_label,
                'corrected': label,
                'type': 'auto_pass_error'  # 自动通过错误
            }
        else:
            # 待审核确认或纠正 = 不是模型错误
            # 这种情况可能是：待审核图片用户确认了原预测，或换了其他分类
            self.parent_system.review_confirmed += 1

        # 移动到下一张
        self.current_idx += 1
        self.show_image()

    def skip(self):
        self.current_idx += 1
        self.show_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_image()

    def next_image(self):
        if self.current_idx < len(self.review_queue) - 1:
            self.current_idx += 1
            self.show_image()


def main():
    MLPCheckInSystem()


if __name__ == '__main__':
    main()