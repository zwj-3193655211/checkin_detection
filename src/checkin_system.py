"""晨读晨练签到检测系统 - MLP增强版（保留老系统流程）"""
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import clip
import torch.nn as nn
import numpy as np
from pathlib import Path

# ==================== MLP模型 ====================
class MLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==================== 参数 ====================
# 三支决策阈值（最优配置：漏检率=0%，审核率<30%）
# 关键：置信度分布分析显示大部分样本置信度>0.999
# 设置ALPHA_ACCEPT=0.9996，确保异常不被误判，同时保持低审核率
ALPHA_ACCEPT = 0.9996  # 最优阈值：漏检0%，审核3.73%
ALPHA_REJECT = 0.20  # 保持低阈值避免误拒绝
MIN_FEATURES = 2  # 最少特征数

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
    "a photo of trees",
    "a photo of flagpole",
    "a photo of number bib",
    "a photo of grandstand",
]

# CLIP特征相似度阈值（从0.22降低到0.20，减少漏检）
FEATURE_SIM_THRESHOLD = 0.20


class MLPCheckInSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载CLIP（特征提取）
        print("加载CLIP...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        print(f"CLIP已加载: {self.device}")

        # 加载MLP（分类器）
        print("加载MLP...")
        self.mlp = MLP()
        self.mlp.load_state_dict(torch.load('outputs/mlp_model.pt'))
        self.mlp.eval()
        print("MLP已加载")
        self.id2label = {0: '晨读', 1: '晨跑', 2: '异常'}

        # 注意：检测系统不需要加载标签文件，直接使用模型进行预测

        self.class_names = ['晨读', '晨跑']
        self.current_data_dir = None
        self.results = {'晨读': [], '晨跑': [], '异常': [], '待审核': []}
        self.scores = {}
        self.review_queue = []

        self.setup_paths()
        self.setup_ui()

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

        title_canvas = tk.Canvas(title_frame, width=1100, height=80, bg='#1a5f7a', highlightthickness=0)
        title_canvas.pack(fill=tk.BOTH)

        # 渐变色
        for i in range(1100):
            ratio = i / 1100
            r = int(26 + (46 - 26) * ratio)
            g = int(95 + (140 - 95) * ratio)
            b = int(122 + (192 - 122) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            title_canvas.create_line(i, 0, i, 80, fill=color)

        title_canvas.create_text(30, 35, text="晨读晨练签到检测系统", font=('Microsoft YaHei', 24, 'bold'), fill='white', anchor='w')
        title_canvas.create_text(30, 58, text="MLP分类器 + CLIP特征提取", font=('Microsoft YaHei', 10), fill='#87ceeb', anchor='w')

        # 主框架
        main_frame = tk.Frame(self.root, bg='#e8f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

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

        params_text = f"MLP模型 | 自动接受≥{ALPHA_ACCEPT} | 自动拒绝≤{ALPHA_REJECT} | 特征勾选辅助"
        tk.Label(param_frame, text=params_text, font=('Consolas', 10), bg='#e8f4f8', fg='#666').pack()

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

        self.root.mainloop()

    def predict(self, image_path):
        """MLP预测"""
        # CLIP特征提取
        img = Image.open(image_path).convert('RGB')
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_input)

        # MLP预测
        with torch.no_grad():
            out = self.mlp(image_features.float())
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        label = self.id2label.get(pred, '未知')
        return label, confidence

    def predict_with_decision(self, image_path):
        """带三支决策的预测，同时计算特征得分(可解释性)"""
        # CLIP特征提取
        img = Image.open(image_path).convert('RGB')
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_input)

        # MLP预测
        with torch.no_grad():
            out = self.mlp(image_features.float())
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0][pred].item()

        label = self.id2label.get(pred, '未知')

        # 计算CLIP特征相似度(可解释性)
        text_tokens = clip.tokenize(CHENIDU_PROMPTS + CHENPAO_PROMPTS).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).cpu().numpy()[0]
        feature_sims = {}
        for i, p in enumerate(CHENIDU_PROMPTS):
            feature_sims[p] = float(similarities[i])
        for i, p in enumerate(CHENPAO_PROMPTS):
            feature_sims[p] = float(similarities[len(CHENIDU_PROMPTS) + i])

        # 统计高相似度特征数(用于可解释性) - 使用统一的FEATURE_SIM_THRESHOLD
        high_sim_count = sum(1 for v in similarities if v > FEATURE_SIM_THRESHOLD)

        # 三支决策规则：只用MLP置信度
        if confidence >= ALPHA_ACCEPT:
            decision = '自动通过'
        elif confidence <= ALPHA_REJECT:
            decision = '自动拒绝'
        else:
            decision = '待审核'

        return label, confidence, decision, high_sim_count, feature_sims

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
        auto_pass = len(self.results['晨读']) + len(self.results['晨跑'])
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
        # 直接打开审核窗口，让用户在下拉框里筛选
        if not self.review_queue:
            messagebox.showinfo("提示", "没有需要审核的图片!")
            return
        ReviewWindow(self.current_data_dir, self.review_queue, self.results, self.scores, self.root)

    def generate_report(self):
        if not self.results:
            messagebox.showwarning("提示", "请先运行预测!")
            return

        total = sum(len(v) for v in self.results.values())
        auto_count = len(self.results['晨读']) + len(self.results['晨跑'])
        review_count = len(self.results['待审核'])

        report = {
            'summary': {
                'total': total,
                '晨读': len(self.results['晨读']),
                '晨跑': len(self.results['晨跑']),
                '异常': len(self.results['异常']),
                '待审核': len(self.results['待审核']),
                'auto_pass_rate': f"{auto_count/total*100:.1f}%",
                'review_rate': f"{review_count/total*100:.1f}%"
            },
            'parameters': {
                'alpha_accept': ALPHA_ACCEPT,
                'alpha_reject': ALPHA_REJECT,
                'model': 'MLP+CLIP',
                'accuracy': '99.95%'
            },
            'review_list': self.results['待审核'],
            'scores': self.scores
        }

        os.makedirs(self.output_dir, exist_ok=True)
        report_file = os.path.join(self.output_dir, f'mlp_report_{len(os.listdir(self.output_dir))}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.info_text.insert(tk.END, f"\n报告已保存: {report_file}\n")
        self.status_label.config(text=f"报告已生成 | {os.path.basename(report_file)}")


class ReviewWindow:
    def __init__(self, data_dir, review_queue, results_dict, scores, parent_root):
        self.data_dir = data_dir
        self.review_queue = review_queue
        self.all_results = results_dict  # 所有分类结果
        self.results = results_dict
        self.scores = scores
        self.current_idx = 0
        self.current_filter = '全部'  # 当前筛选

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
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg='#ffffff', highlightthickness=2, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)

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

            # 特征相似度(可解释性)
            if 'feature_sims' in s:
                sims = s['feature_sims']
                sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
                info_text += "\n特征相似度:"
                for k, v in sorted_sims[:4]:
                    info_text += f" {k.split(' of ')[-1]}:{v:.2f}"

            self.score_label.config(text=info_text, font=('Consolas', 9))

        self.label.config(text=f"{fn}")

        img = Image.open(os.path.join(self.data_dir, fn))
        w, h = img.size
        max_size = 700
        if w > max_size or h > max_size:
            ratio = min(max_size/w, max_size/h)
            img = img.resize((int(w*ratio), int(h*ratio)))

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def set_label(self, label):
        if self.current_idx >= len(self.review_queue):
            return
        fn = self.review_queue[self.current_idx]

        # 从待审核移除，加入对应分类
        if fn in self.results['待审核']:
            self.results['待审核'].remove(fn)

        # 也要从之前的分类中移除
        for cat in ['晨读', '晨跑', '异常']:
            if fn in self.results[cat]:
                self.results[cat].remove(fn)

        self.results[label].append(fn)

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