"""
特征标注工具（纯人工版）

功能:
- 先选主类别，再勾选特征
- 晨读特征：人物、座椅、教室、投影幕布
- 晨跑特征：操场、天空、人物、绿地、旗杆、号码牌
"""

import os
import json
import random
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class FeatureLabelTool:
    """特征标注工具"""

    # 主标签选项
    MAIN_LABELS = ["晨读", "晨跑", "异常"]

    # 晨读特征
    CHENREAD_FEATURES = ["人脸", "蓝色桌子", "教室", "投影幕布"]

    # 晨跑特征
    CHENPAO_FEATURES = ["人脸", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]

    # 异常原因
    ABNORMAL_REASONS = ["太暗", "没有人", "场景不对", "背景模糊"]

    # CLIP特征检测阈值（分特征设置）
    # 策略：降低阈值减少漏检，提高召回率
    FEATURE_THRESHOLDS = {
        # 晨读特征
        "人脸": 0.20,      # 降低0.02，减少漏检
        "蓝色桌子": 0.22,  # 降低0.03，可能与其他蓝色混淆
        "教室": 0.22,     # 降低0.03，泛化较强
        "投影幕布": 0.18,  # 降低0.02，特征明显容易检测

        # 晨跑特征
        "人脸": 0.20,
        "跑道": 0.20,      # 降低0.02，颜色特征明显
        "天空": 0.16,      # 降低0.02，容易检测
        "绿地": 0.16,      # 降低0.02，容易检测
        "树木": 0.20,
        "旗杆": 0.13,     # 降低0.02，容易漏检
        "号码布": 0.13,    # 降低0.02，容易漏检
        "主席台": 0.18,
    }

    def __init__(self, picture_dir: str, output_file: str = None):
        self.picture_dir = Path(picture_dir)
        self.output_file = output_file or str(self.picture_dir.parent / "data" / "labels.json")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        self.image_files = self._get_image_files()

        random.seed(42)
        self.original_order = self.image_files.copy()
        random.shuffle(self.image_files)

        self.labels = self._load_labels()

        # 加载MLP和CLIP用于预标注
        self.mlp = None
        self.clip_model = None
        self.preprocess = None
        self._load_models()

        self.current_idx = 0
        self.filter_mode = None

        # 初始化 tkinter
        self.root = tk.Tk()
        self.root.title("晨读晨练特征标注工具 (MLP预填)")
        self.root.geometry("1000x850")

        self.main_label_var = tk.StringVar(value="晨读")
        self.feature_vars = {}
        self.abnormal_var = tk.StringVar()

        self._create_widgets()

        if self.image_files:
            self._show_image()

    def _load_models(self):
        """加载MLP和CLIP模型用于预标注"""
        try:
            import torch
            import clip
            import torch.nn as nn

            # 加载CLIP
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            self.device = device

            # 特征提示词
            self.FEATURE_PROMPTS = {
                "晨读": {
                    "人脸": "a photo of a human face",
                    "蓝色桌子": "a photo of blue desks",
                    "教室": "a photo of classroom",
                    "投影幕布": "a photo of projection screen",
                },
                "晨跑": {
                    "人脸": "a photo of a human face",
                    "跑道": "a photo of running track",
                    "天空": "a photo of blue sky",
                    "绿地": "a photo of green grass",
                    "树木": "a photo of trees",
                    "旗杆": "a photo of flagpole",
                    "号码布": "a photo of number bib",
                    "主席台": "a photo of grandstand",
                }
            }

            # 加载MLP
            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 3))
                def forward(self, x):
                    return self.net(x)

            self.mlp = MLP()
            model_path = self.picture_dir.parent / "cache" / "mlp_model.pt"
            if model_path.exists():
                self.mlp.load_state_dict(torch.load(model_path))
                self.mlp.eval()
                print(f"MLP+CLIP已加载，使用设备: {device}")
        except Exception as e:
            print(f"模型加载失败: {e}")

    def _get_image_files(self) -> list:
        files = []
        for f in os.listdir(self.picture_dir):
            if f.lower().endswith(self.image_extensions):
                files.append(f)
        return sorted(files)

    def _load_labels(self) -> dict:
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'labels' in data:
                        return data['labels']
                    return data
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_labels(self):
        data = {
            "_schema": "checkin_labels_v2",
            "labels": self.labels,
            "metadata": {
                "updated": self._get_timestamp(),
                "total_images": len(self.image_files),
                "labeled_count": len(self.labels),
            }
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _create_widgets(self):
        # 顶部信息栏
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(
            info_frame,
            text="进度: 0 / 0 (已标注: 0)",
            font=('Arial', 12)
        )
        self.progress_label.pack(side=tk.LEFT)

        self.stats_label = ttk.Label(
            info_frame,
            text="",
            font=('Arial', 10)
        )
        self.stats_label.pack(side=tk.RIGHT)

        # 图片显示区域
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.filename_label = ttk.Label(
            self.root,
            text="",
            font=('Arial', 11)
        )
        self.filename_label.pack(pady=2)

        # ===== 主类别选择 =====
        main_label_frame = ttk.Frame(self.root)
        main_label_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(main_label_frame, text="主类别:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=5)

        for label in self.MAIN_LABELS:
            ttk.Radiobutton(
                main_label_frame,
                text=label,
                variable=self.main_label_var,
                value=label,
                command=self._on_main_label_change
            ).pack(side=tk.LEFT, padx=10)

        # ===== 特征选择区域 =====
        self.feature_frame = ttk.LabelFrame(self.root, text="特征勾选", padding=10)
        self.feature_frame.pack(fill=tk.X, padx=10, pady=5)

        # 晨读特征
        self.chenread_feature_frame = ttk.Frame(self.feature_frame)
        self.chenread_feature_frame.pack(fill=tk.X)

        ttk.Label(self.chenread_feature_frame, text="晨读特征:").pack(side=tk.LEFT, padx=5)

        for feature in self.CHENREAD_FEATURES:
            var = tk.BooleanVar(value=False)
            self.feature_vars[f"晨读_{feature}"] = var
            ttk.Checkbutton(
                self.chenread_feature_frame,
                text=feature,
                variable=var
            ).pack(side=tk.LEFT, padx=8)

        # 晨跑特征
        self.chenpao_feature_frame = ttk.Frame(self.feature_frame)
        self.chenpao_feature_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.chenpao_feature_frame, text="晨跑特征:").pack(side=tk.LEFT, padx=5)

        for feature in self.CHENPAO_FEATURES:
            var = tk.BooleanVar(value=False)
            self.feature_vars[f"晨跑_{feature}"] = var
            ttk.Checkbutton(
                self.chenpao_feature_frame,
                text=feature,
                variable=var
            ).pack(side=tk.LEFT, padx=8)

        # 异常原因
        self.abnormal_frame = ttk.Frame(self.feature_frame)
        self.abnormal_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.abnormal_frame, text="异常原因:").pack(side=tk.LEFT, padx=5)

        for reason in self.ABNORMAL_REASONS:
            ttk.Radiobutton(
                self.abnormal_frame,
                text=reason,
                variable=self.abnormal_var,
                value=reason
            ).pack(side=tk.LEFT, padx=8)

        # ===== 按钮区域 =====
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill=tk.X)

        ttk.Button(row1_frame, text="上一张 (↑)", command=self.prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1_frame, text="下一张 (↓)", command=self.next).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1_frame, text="跳过 (K)", command=self.skip).pack(side=tk.LEFT, padx=3)

        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill=tk.X, pady=3)

        ttk.Button(row2_frame, text="跳转到 (G)", command=self.goto_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2_frame, text="随机跳转 (R)", command=self.random_goto).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2_frame, text="未标注 (U)", command=self.goto_first_unlabeled).pack(side=tk.LEFT, padx=3)

        row3_frame = ttk.Frame(button_frame)
        row3_frame.pack(fill=tk.X, pady=3)

        ttk.Label(row3_frame, text="筛选:").pack(side=tk.LEFT, padx=3)
        self.filter_combo = ttk.Combobox(
            row3_frame,
            values=['全部', '未标注', '晨读', '晨跑', '异常'],
            state='readonly',
            width=10
        )
        self.filter_combo.set('全部')
        self.filter_combo.pack(side=tk.LEFT, padx=3)
        self.filter_combo.bind('<<ComboboxSelected>>', lambda e: self._on_filter_change())

        # 状态栏
        self.status_label = ttk.Label(
            self.root,
            text="就绪",
            font=('Arial', 9),
            foreground="green"
        )
        self.status_label.pack(pady=5)

        # 绑定快捷键
        self.root.bind('<Up>', lambda e: self.prev())
        self.root.bind('<Down>', lambda e: self.next())
        self.root.bind('<k>', lambda e: self.skip())
        self.root.bind('<g>', lambda e: self.goto_image())
        self.root.bind('<r>', lambda e: self.random_goto())
        self.root.bind('<u>', lambda e: self.goto_first_unlabeled())

        # 数字键快捷标注主类别
        for i, label in enumerate(self.MAIN_LABELS):
            self.root.bind(f'{i+1}', lambda e, l=label: self._set_main_label(l))

        self.root.focus_force()

    def _on_main_label_change(self):
        main_label = self.main_label_var.get()

        if main_label == "晨读":
            self.chenread_feature_frame.pack(fill=tk.X)
            self.chenpao_feature_frame.pack_forget()
            self.abnormal_frame.pack_forget()
        elif main_label == "晨跑":
            self.chenread_feature_frame.pack_forget()
            self.chenpao_feature_frame.pack(fill=tk.X, pady=5)
            self.abnormal_frame.pack_forget()
        else:  # 异常
            self.chenread_feature_frame.pack_forget()
            self.chenpao_feature_frame.pack_forget()
            self.abnormal_frame.pack(fill=tk.X, pady=5)

    def _set_main_label(self, label: str):
        self.main_label_var.set(label)
        self._on_main_label_change()
        self._load_feature_vars()

    def _auto_predict(self, filepath):
        """用MLP预测主类别，CLIP相似度预填特征"""
        print(f"DEBUG: _auto_predict called for {filepath}")
        print(f"DEBUG: clip_model exists: {self.clip_model is not None}")
        print(f"DEBUG: FEATURE_PROMPTS exists: {hasattr(self, 'FEATURE_PROMPTS')}")

        if not self.clip_model:
            print("DEBUG: clip_model is None, skipping")
            return

        try:
            import torch
            import clip

            # CLIP特征提取
            img = Image.open(filepath).convert('RGB')
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_features = self.clip_model.encode_image(img_input)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # MLP预测主类别
            main_label = "晨读"
            if self.mlp:
                with torch.no_grad():
                    out = self.mlp(img_features)
                    probs = torch.softmax(out, dim=1)
                    pred = probs.argmax(dim=1).item()
                    main_label = ["晨读", "晨跑", "异常"][pred]

            print(f"DEBUG: predicted main_label = {main_label}")

            self.main_label_var.set(main_label)
            self._on_main_label_change()

            # 重置特征勾选
            for var in self.feature_vars.values():
                var.set(False)

            # CLIP相似度预填特征
            prompts = self.FEATURE_PROMPTS.get(main_label, {})
            print(f"DEBUG: prompts = {prompts}")
            prompt_list = list(prompts.values())

            if prompt_list:
                text_tokens = clip.tokenize(prompt_list).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarities = (img_features @ text_features.T).cpu().numpy()[0]

                print(f"DEBUG: similarities = {similarities}")

                # 使用分特征动态阈值
                for i, feat_name in enumerate(prompts.keys()):
                    key = f"{main_label}_{feat_name}"
                    if key in self.feature_vars:
                        # 获取该特征的阈值
                        threshold = self.FEATURE_THRESHOLDS.get(feat_name, 0.2)
                        result = bool(similarities[i] > threshold)
                        print(f"DEBUG: {feat_name} sim={similarities[i]:.3f} thresh={threshold:.2f} -> {result}")
                        self.feature_vars[key].set(result)

        except Exception as e:
            print(f"预测失败: {e}")

    def _load_feature_vars(self):
        filename = self.image_files[self.current_idx]

        for var in self.feature_vars.values():
            var.set(False)
        self.abnormal_var.set("")

        if filename in self.labels:
            info = self.labels[filename]
            main_label = info.get('label', '晨读')
            self.main_label_var.set(main_label)
            self._on_main_label_change()

            if 'features' in info:
                prefix = f"{main_label}_"
                for f, v in info['features'].items():
                    key = prefix + f
                    if key in self.feature_vars:
                        self.feature_vars[key].set(v)

            if 'reason' in info:
                self.abnormal_var.set(info['reason'])

    def _save_current_label(self):
        filename = self.image_files[self.current_idx]
        main_label = self.main_label_var.get()

        label_data = {
            'label': main_label,
            'scene': 'morning_reading' if main_label == '晨读' else 'morning_running' if main_label == '晨跑' else 'abnormal',
            'is_normal': 'normal' if main_label != '异常' else 'abnormal',
            'timestamp': self._get_timestamp()
        }

        if main_label == '晨读':
            features = {}
            for feature in self.CHENREAD_FEATURES:
                features[feature] = self.feature_vars[f"晨读_{feature}"].get()
            label_data['features'] = features
        elif main_label == '晨跑':
            features = {}
            for feature in self.CHENPAO_FEATURES:
                features[feature] = self.feature_vars[f"晨跑_{feature}"].get()
            label_data['features'] = features
        elif main_label == '异常':
            reason = self.abnormal_var.get()
            if reason:
                label_data['reason'] = reason

        self.labels[filename] = label_data
        self._save_labels()

        self.status_label.config(text=f"已保存: {main_label}", foreground="green")
        self._update_progress()

    def _show_image(self):
        self.root.focus_set()

        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("完成", "所有图片已标注完成！")
            return

        filename = self.image_files[self.current_idx]
        filepath = os.path.join(self.picture_dir, filename)

        try:
            img = Image.open(filepath)

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width < 100 or canvas_height < 100:
                canvas_width = 900
                canvas_height = 600

            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            img_display = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(img_display)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER)

            self.filename_label.config(text=f"{filename}")

            # 加载已有标注或用MLP+CLIP预填
            has_label = filename in self.labels
            has_features = has_label and self.labels[filename].get('features')

            # 只在完全没有任何标注时才使用MLP预填
            # 如果已有主类别标注，即使没有features也不覆盖
            if not has_label:
                # 完全新图片，用MLP+CLIP预填
                self._auto_predict(filepath)
                self.status_label.config(text=f"MLP预填: {self.main_label_var.get()}", foreground="orange")
            else:
                # 已有标注，加载已有内容
                self._load_feature_vars()
                self.status_label.config(text=f"已有标注: {self.labels[filename].get('label')}", foreground="blue")

            self._update_progress()

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {e}")
            self.skip()

    def _update_progress(self):
        labeled_count = len(self.labels)
        total = len(self.image_files)

        self.progress_label.config(
            text=f"进度: {self.current_idx + 1} / {total} (已标注: {labeled_count})"
        )

        label_counts = {l: 0 for l in self.MAIN_LABELS}
        for info in self.labels.values():
            label = info.get('label', '不确定')
            if label in label_counts:
                label_counts[label] += 1

        self.stats_label.config(
            text=f"晨读:{label_counts['晨读']} 晨跑:{label_counts['晨跑']} 异常:{label_counts['异常']}"
        )

    def _on_filter_change(self):
        filter_type = self.filter_combo.get()

        if filter_type == '全部':
            self.filter_mode = None
            self.image_files = self.original_order.copy()
        elif filter_type == '未标注':
            self.filter_mode = 'unlabeled'
            self.image_files = [f for f in self.original_order if f not in self.labels]
        else:
            self.filter_mode = filter_type
            self.image_files = [
                f for f in self.original_order
                if f in self.labels and self.labels[f].get('label') == filter_type
            ]

        random.shuffle(self.image_files)
        self.current_idx = 0

        if self.image_files:
            self._show_image()
        else:
            messagebox.showinfo("提示", f"没有找到符合条件的图片")
            self.filter_combo.set('全部')
            self.filter_mode = None
            self.image_files = self.original_order.copy()
            random.shuffle(self.image_files)
            self.current_idx = 0
            self._show_image()

    def next(self):
        self._save_current_label()
        self._move_to_next()

    def _move_to_next(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self._show_image()
        else:
            messagebox.showinfo("完成", "所有图片已标注完成！")

    def prev(self):
        self._save_current_label()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_image()

    def skip(self):
        self._save_current_label()
        self.current_idx += 1
        self._show_image()

    def goto_image(self):
        input_str = simpledialog.askstring("跳转", f"索引 (1-{len(self.image_files)}):")
        if input_str:
            try:
                idx = int(input_str) - 1
                if 0 <= idx < len(self.image_files):
                    self.current_idx = idx
                    self._show_image()
                else:
                    messagebox.showerror("错误", "索引超出范围")
            except ValueError:
                messagebox.showerror("错误", "请输入有效数字")

    def random_goto(self):
        self.current_idx = random.randint(0, len(self.image_files) - 1)
        self._show_image()

    def goto_first_unlabeled(self):
        for i, filename in enumerate(self.image_files):
            if filename not in self.labels:
                self.current_idx = i
                self._show_image()
                return
        messagebox.showinfo("提示", "所有图片都已标注！")

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="特征标注工具")
    parser.add_argument('--picture_dir', type=str,
                        default=str(Path(__file__).parent.parent / "data" / "raw"),
                        help='图片目录')
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).parent.parent / "data" / "labels.json"),
                        help='输出文件')

    args = parser.parse_args()
    tool = FeatureLabelTool(args.picture_dir, args.output)
    tool.run()


if __name__ == '__main__':
    main()