"""
标注工具增强版

功能:
- 支持显示标注进度
- 快捷键导航
- 批量标注
- 跳转到指定图片
- 统计摘要
"""

import os
import json
import random
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class EnhancedLabelTool:
    """
    增强版标注工具
    """

    # 标签选项
    LABELS = ["晨读", "晨跑", "异常", "不确定"]

    # 标签到场景和异常的映射
    LABEL_TO_SCENE = {
        "晨读": "morning_reading",
        "晨跑": "morning_running",
        "异常": "abnormal",
        "不确定": "undecided"
    }

    LABEL_TO_NORMAL = {
        "晨读": "normal",
        "晨跑": "normal",
        "异常": "abnormal",
        "不确定": "undecided"
    }

    def __init__(self, picture_dir: str, output_file: str = None):
        """
        Args:
            picture_dir: 图片目录
            output_file: 标签输出文件路径
        """
        self.picture_dir = Path(picture_dir)
        self.output_file = output_file or str(self.picture_dir.parent / "data" / "labels.json")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # 获取所有图片
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        self.image_files = self._get_image_files()

        # 打乱顺序
        random.seed(42)
        self.original_order = self.image_files.copy()
        random.shuffle(self.image_files)

        # 加载已有标签
        self.labels = self._load_labels()

        # 当前索引
        self.current_idx = 0

        # 筛选模式: None=全部, 'unlabeled'=只显示未标注
        self.filter_mode = None

        # 初始化 tkinter
        self.root = tk.Tk()
        self.root.title("晨读晨练签到标注工具")
        self.root.geometry("900x700")

        # 创建 StringVar 在窗口创建之后
        self.filter_var = tk.StringVar(value="全部")

        self._create_widgets()

        # 显示第一张
        if self.image_files:
            self._show_image()

    def _get_image_files(self) -> list:
        """获取所有图片文件"""
        files = []
        for f in os.listdir(self.picture_dir):
            if f.lower().endswith(self.image_extensions):
                files.append(f)
        return sorted(files)

    def _load_labels(self) -> dict:
        """加载标签文件"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 支持新旧格式
                    if 'labels' in data:
                        return data['labels']
                    return data
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_labels(self):
        """保存标签"""
        data = {
            "_schema": "checkin_labels_v1",
            "labels": self.labels,
            "metadata": {
                "created": "2026-04-22",
                "updated": self._get_timestamp(),
                "total_images": len(self.image_files),
                "labeled_count": len(self.labels),
                "label_options": self.LABELS
            }
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _create_widgets(self):
        """创建界面组件"""

        # 顶部信息栏
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        # 进度信息
        self.progress_label = ttk.Label(
            info_frame,
            text="进度: 0 / 0 (已标注: 0)",
            font=('Arial', 12)
        )
        self.progress_label.pack(side=tk.LEFT)

        # 统计信息
        self.stats_label = ttk.Label(
            info_frame,
            text="晨读:0 晨跑:0 异常:0 不确定:0",
            font=('Arial', 10)
        )
        self.stats_label.pack(side=tk.RIGHT)

        # 图片显示区域
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 文件名和路径
        self.filename_label = ttk.Label(
            self.root,
            text="",
            font=('Arial', 11)
        )
        self.filename_label.pack(pady=2)

        # 标签选择
        label_frame = ttk.Frame(self.root)
        label_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(label_frame, text="标签:").pack(side=tk.LEFT, padx=5)

        self.label_var = tk.StringVar(value="晨读")
        for label in self.LABELS:
            ttk.Radiobutton(
                label_frame,
                text=label,
                variable=self.label_var,
                value=label,
                command=self._on_label_change
            ).pack(side=tk.LEFT, padx=10)

        # 按钮区域
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        # 第一行按钮
        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill=tk.X)

        ttk.Button(row1_frame, text="上一张 (↑/P)", command=self.prev).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1_frame, text="下一张 (↓/N)", command=self.next).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1_frame, text="跳过 (K)", command=self.skip).pack(side=tk.LEFT, padx=3)

        # 第二行按钮
        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill=tk.X, pady=3)

        ttk.Button(row2_frame, text="跳转到... (G)", command=self.goto_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2_frame, text="随机跳转 (R)", command=self.random_goto).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2_frame, text="未标注首张 (U)", command=self.goto_first_unlabeled).pack(side=tk.LEFT, padx=3)

        # 第三行按钮
        row3_frame = ttk.Frame(button_frame)
        row3_frame.pack(fill=tk.X, pady=3)

        # 筛选下拉框
        ttk.Label(row3_frame, text="筛选:").pack(side=tk.LEFT, padx=3)
        self.filter_combo = ttk.Combobox(
            row3_frame,
            values=['全部', '未标注', '已标注', '晨跑', '晨读', '异常', '不确定'],
            state='readonly',
            width=12
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
        self.status_label.pack(pady=2)

        # 绑定快捷键
        self.root.bind('<Up>', lambda e: self.prev())
        self.root.bind('<Down>', lambda e: self.next())
        self.root.bind('<Left>', lambda e: self.prev())
        self.root.bind('<Right>', lambda e: self.next())
        self.root.bind('<p>', lambda e: self.prev())
        self.root.bind('<n>', lambda e: self.next())
        self.root.bind('<k>', lambda e: self.skip())
        self.root.unbind('<Control-s>')
        self.root.bind('<g>', lambda e: self.goto_image())
        self.root.bind('<r>', lambda e: self.random_goto())
        self.root.bind('<u>', lambda e: self.goto_first_unlabeled())
        self.root.bind('<Control-s>', lambda e: self.save_all())

        # 键盘数字键快速标注
        for i, label in enumerate(self.LABELS):
            self.root.bind(f'{i+1}', lambda e, l=label: self._quick_label(l))

        # 让窗口获取焦点，确保快捷键生效
        self.root.focus_force()

    def _on_label_change(self):
        """标签改变时的回调"""
        pass

    def _on_filter_change(self):
        """筛选条件改变时的回调"""
        filter_type = self.filter_combo.get()

        if filter_type == '全部':
            self.filter_mode = None
            self.image_files = self.original_order.copy()
        elif filter_type == '未标注':
            self.filter_mode = 'unlabeled'
            self.image_files = [f for f in self.original_order if f not in self.labels]
        elif filter_type == '已标注':
            self.filter_mode = 'labeled'
            self.image_files = [f for f in self.original_order if f in self.labels]
        else:
            # 按标签筛选
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
            # 切回全部
            self.filter_combo.set('全部')
            self.filter_mode = None
            self.image_files = self.original_order.copy()
            random.shuffle(self.image_files)
            self.current_idx = 0
            self._show_image()

        self.status_label.config(text=f"筛选: {filter_type}", foreground="purple")

    def _quick_label(self, label: str):
        """快速标注并下一张"""
        self.label_var.set(label)
        self._save_current_label()
        self.current_idx += 1
        self._show_image()

    def _save_current_label(self):
        """保存当前标签"""
        filename = self.image_files[self.current_idx]
        label = self.label_var.get()

        self.labels[filename] = {
            'label': label,
            'scene': self.LABEL_TO_SCENE[label],
            'is_normal': self.LABEL_TO_NORMAL[label],
            'timestamp': self._get_timestamp()
        }

        self._save_labels()
        self.status_label.config(text=f"已保存: {label}", foreground="green")

    def _show_image(self):
        """显示当前图片"""
        # 确保窗口获取焦点，快捷键才能生效
        self.root.focus_set()

        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("完成", "所有图片已标注完成！")
            return

        filename = self.image_files[self.current_idx]
        filepath = os.path.join(self.picture_dir, filename)

        # 加载并显示图片
        try:
            img = Image.open(filepath)

            # 计算缩放比例以适应窗口
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # 确保 canvas 尺寸有效（窗口可能还没完成渲染）
            if canvas_width < 100 or canvas_height < 100:
                canvas_width = 800
                canvas_height = 500

            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            img_display = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 转换为tkinter可用的格式
            self.photo = ImageTk.PhotoImage(img_display)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER)

            # 更新文件名
            self.filename_label.config(text=f"{filename}")

            # 如果已有标签，恢复界面状态
            if filename in self.labels:
                label = self.labels[filename].get('label', '晨读')
                self.label_var.set(label)
                self.status_label.config(text=f"已有标注: {label}", foreground="blue")
            else:
                self.status_label.config(text="未标注", foreground="gray")

            # 更新进度
            self._update_progress()

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {e}")
            self.skip()

    def _update_progress(self):
        """更新进度显示"""
        labeled_count = len(self.labels)
        total = len(self.image_files)

        self.progress_label.config(
            text=f"进度: {self.current_idx + 1} / {total} (已标注: {labeled_count})"
        )

        # 统计各标签数量
        label_counts = {l: 0 for l in self.LABELS}
        for info in self.labels.values():
            label = info.get('label', '不确定')
            if label in label_counts:
                label_counts[label] += 1

        self.stats_label.config(
            text=f"晨读:{label_counts['晨读']} 晨跑:{label_counts['晨跑']} 异常:{label_counts['异常']} 不确定:{label_counts['不确定']}"
        )

    def save_and_next(self):
        """保存当前标签并移到下一张"""
        filename = self.image_files[self.current_idx]
        label = self.label_var.get()

        # 保存标签
        self.labels[filename] = {
            'label': label,
            'scene': self.LABEL_TO_SCENE[label],
            'is_normal': self.LABEL_TO_NORMAL[label],
            'timestamp': self._get_timestamp()
        }

        self._save_labels()
        self.status_label.config(text=f"已保存: {label}", foreground="green")

        # 移到下一张
        self.current_idx += 1
        self._show_image()

    def save_current(self):
        """保存当前标注并移到下一张"""
        filename = self.image_files[self.current_idx]
        label = self.label_var.get()

        self.labels[filename] = {
            'label': label,
            'scene': self.LABEL_TO_SCENE[label],
            'is_normal': self.LABEL_TO_NORMAL[label],
            'timestamp': self._get_timestamp()
        }

        self._save_labels()
        self.status_label.config(text=f"已保存: {label}", foreground="green")

        # 移到下一张
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self._show_image()

    def next(self):
        """下一张"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self._show_image()

    def prev(self):
        """上一张"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_image()

    def skip(self):
        """跳过当前图片"""
        self.current_idx += 1
        self._show_image()

    def goto_image(self):
        """跳转到指定索引"""
        input_str = simpledialog.askstring("跳转到", f"请输入图片索引 (1-{len(self.image_files)}):")
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
        """随机跳转"""
        self.current_idx = random.randint(0, len(self.image_files) - 1)
        self._show_image()

    def goto_first_unlabeled(self):
        """跳转到第一张未标注的图片"""
        for i, filename in enumerate(self.image_files):
            if filename not in self.labels:
                self.current_idx = i
                self._show_image()
                return
        messagebox.showinfo("提示", "所有图片都已标注！")

    def filter_unlabeled(self):
        """切换只显示未标注的图片"""
        if self.filter_mode == 'unlabeled':
            # 已经是非标注模式，切换回全部
            self.filter_mode = None
            self.image_files = self.original_order.copy()
            random.shuffle(self.image_files)
            self.current_idx = 0
            self._show_image()
            self.status_label.config(text="显示全部", foreground="green")
        else:
            # 切换到只显示未标注
            self.filter_mode = 'unlabeled'
            self.image_files = [f for f in self.original_order if f not in self.labels]
            random.shuffle(self.image_files)
            self.current_idx = 0
            self._show_image()
            self.status_label.config(text="只看未标注 ✨", foreground="orange")

    def filter_all(self):
        """显示所有图片"""
        self.filter_mode = None
        self.image_files = self.original_order.copy()
        random.shuffle(self.image_files)
        self.current_idx = 0
        self._show_image()
        self.status_label.config(text="显示全部", foreground="green")

    def export_stats(self):
        """导出统计信息"""
        labeled_count = len(self.labels)

        label_counts = {l: 0 for l in self.LABELS}
        for info in self.labels.values():
            label = info.get('label', '不确定')
            if label in label_counts:
                label_counts[label] += 1

        stats_text = f"""
标注统计
========
总数: {len(self.image_files)}
已标注: {labeled_count}
未标注: {len(self.image_files) - labeled_count}

标签分布:
- 晨读: {label_counts['晨读']}
- 晨跑: {label_counts['晨跑']}
- 异常: {label_counts['异常']}
- 不确定: {label_counts['不确定']}
        """

        messagebox.showinfo("统计信息", stats_text.strip())

    def save_all(self):
        """保存所有标签"""
        self._save_labels()
        messagebox.showinfo("保存", f"已保存 {len(self.labels)} 个标签")

    def run(self):
        """运行标注工具"""
        self.root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="签到图片标注工具 (增强版)")
    parser.add_argument('--picture_dir', type=str,
                       default=str(Path(__file__).parent.parent / "data" / "raw"),
                       help='图片目录')
    parser.add_argument('--output', type=str, 
                       default=str(Path(__file__).parent.parent / "data" / "labels.json"), 
                       help='输出文件')

    args = parser.parse_args()

    tool = EnhancedLabelTool(args.picture_dir, args.output)
    tool.run()


if __name__ == '__main__':
    main()
