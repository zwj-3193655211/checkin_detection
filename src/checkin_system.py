"""晨读晨练签到检测系统 - 完整流程"""
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import sys
sys.path.append(os.path.dirname(__file__))
from models.three_way_decision import ThreeWayDecision


class CheckInSystem:
    def __init__(self):
        self.model = None
        self.transform = None
        self.class_names = ['晨读', '晨跑', '异常']
        self.current_data_dir = None
        self.results = {}
        self.review_queue = []
        self.three_way_decision = None
        self.alpha = 0.85
        self.beta = 0.35

        self.setup_path()
        self.load_model()
        self.load_three_way_thresholds()
        self.setup_ui()
    
    def setup_path(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base, 'outputs', 'resnet18_best.pt')
        self.output_dir = os.path.join(base, 'outputs')
    
    def load_model(self):
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_three_way_thresholds(self):
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            if 'three_way_decision' in report:
                self.alpha = report['three_way_decision']['alpha']
                self.beta = report['three_way_decision']['beta']
                self.three_way_decision = ThreeWayDecision(alpha=self.alpha, beta=self.beta)
                print(f"✅ 已加载三支决策阈值: alpha={self.alpha}, beta={self.beta}")
            else:
                self.three_way_decision = ThreeWayDecision(alpha=self.alpha, beta=self.beta)
                print(f"⚠️ 报告中无三支决策阈值，使用默认值: alpha={self.alpha}, beta={self.beta}")
        else:
            self.three_way_decision = ThreeWayDecision(alpha=self.alpha, beta=self.beta)
            print(f"⚠️ 未找到评估报告，使用默认阈值: alpha={self.alpha}, beta={self.beta}")
    
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("晨读晨练签到检测系统")
        self.root.geometry("1100x800")
        self.root.configure(bg='#e8f4f8')
        
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Microsoft YaHei', 24, 'bold'), background='#e8f4f8', foreground='#1a5f7a')
        style.configure('Info.TLabel', font=('Microsoft YaHei', 11), background='#e8f4f8')
        style.configure('Action.TButton', font=('Microsoft YaHei', 11), padding=10)
        
        title_frame = tk.Frame(self.root, height=80, bg='#1a5f7a')
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_canvas = tk.Canvas(title_frame, width=1100, height=80, bg='#1a5f7a', highlightthickness=0)
        title_canvas.pack(fill=tk.BOTH)
        
        for i in range(1100):
            ratio = i / 1100
            r = int(26 + (46 - 26) * ratio)
            g = int(95 + (140 - 95) * ratio)
            b = int(122 + (192 - 122) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            title_canvas.create_line(i, 0, i, 80, fill=color)
        
        title_canvas.create_text(30, 35, text="🎯 晨读晨练签到检测系统", 
                                 font=('Microsoft YaHei', 24, 'bold'), 
                                 fill='white', anchor='w')
        title_canvas.create_text(30, 58, text="基于深度学习的智能检测平台", 
                                 font=('Microsoft YaHei', 10), 
                                 fill='#87ceeb', anchor='w')
        
        main_frame = tk.Frame(self.root, bg='#e8f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        
        step_frame = tk.LabelFrame(main_frame, text=" 📋 操作步骤 ", font=('Microsoft YaHei', 13, 'bold'),
                                   bg='#e8f4f8', fg='#1a5f7a', padx=20, pady=15,
                                   labelanchor='nw')
        step_frame.pack(fill=tk.X, pady=(0, 20))
        
        btn_frame = tk.Frame(step_frame, bg='#e8f4f8')
        btn_frame.pack(pady=15)
        
        buttons = [
            ("1️ 选择数据目录", self.select_folder, '#3498db'),
            ("2️ 开始判别", self.run_prediction, '#27ae60'),
            ("3️ 人工审核", self.start_review, '#e74c3c'),
            ("4️ 生成报告", self.generate_report, '#9b59b6')
        ]
        
        for i, (text, cmd, color) in enumerate(buttons):
            btn = tk.Button(btn_frame, text=text, command=cmd, width=18, height=2,
                           font=('Microsoft YaHei', 12, 'bold'), bg=color, fg='white',
                           activebackground=color, activeforeground='white',
                           relief=tk.RAISED, cursor='hand2', bd=0,
                           highlightthickness=0)
            btn.pack(side=tk.LEFT, padx=10, ipady=5)
            btn.bind('<Enter>', lambda e, b=btn: b.config(relief=tk.SUNKEN))
            btn.bind('<Leave>', lambda e, b=btn: b.config(relief=tk.RAISED))
        
        result_frame = tk.LabelFrame(main_frame, text=" 📈 检测结果 ", font=('Microsoft YaHei', 13, 'bold'),
                                     bg='#e8f4f8', fg='#1a5f7a', padx=20, pady=15,
                                     labelanchor='nw')
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        result_inner = tk.Frame(result_frame, bg='white', bd=2, relief=tk.SUNKEN)
        result_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(result_inner, width=90, height=22, font=('Consolas', 11),
                                 bg='#fafafa', fg='#2c3e50', relief=tk.FLAT,
                                 padx=15, pady=15, spacing1=3, spacing2=2)
        self.info_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(result_inner, command=self.info_text.yview, style='Modern.Vertical.TScrollbar')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        style.configure('Modern.Vertical.TScrollbar', troughcolor='#e8f4f8', background='#3498db')
        
        status_frame = tk.Frame(self.root, bg='#1a5f7a', height=35)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame,
                                     text=f"✅ 就绪 | 模型: ResNet18 | 三支决策: α={self.alpha:.2f} β={self.beta:.2f}",
                                     font=('Microsoft YaHei', 10), bg='#1a5f7a', fg='white')
        self.status_label.pack(pady=8)
        
        self.root.mainloop()
    
    def select_folder(self):
        folder = filedialog.askdirectory(title="选择数据包文件夹")
        if folder:
            self.current_data_dir = folder
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.info_text.insert(tk.END, f"\n{'='*60}\n")
            self.info_text.insert(tk.END, f"  已选择文件夹: {folder}\n")
            self.info_text.insert(tk.END, f"  图片数量: {len(files)} 张\n")
            self.info_text.insert(tk.END, f"{'='*60}\n")
            self.status_label.config(text=f"已选择: {os.path.basename(folder)} | 图片: {len(files)}张")
    
    def run_prediction(self):
        if not self.current_data_dir:
            messagebox.showwarning("提示", "请先选择数据包文件夹!")
            return
        
        self.results = {'晨读': [], '晨跑': [], '异常': [], '不确定': []}
        
        files = [f for f in os.listdir(self.current_data_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.info_text.insert(tk.END, f"\n开始预测 {len(files)} 张图片...\n")
        
        for i, fn in enumerate(files):
            if i % 50 == 0:
                self.info_text.insert(tk.END, f"  进度: {i}/{len(files)}\n")
                self.info_text.see(tk.END)
                self.root.update()
            
            try:
                img = Image.open(os.path.join(self.current_data_dir, fn)).convert('RGB')
                img_t = self.transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    out = self.model(img_t)
                    prob = torch.softmax(out, dim=1)

                    normal_prob = prob[0, 0].item() + prob[0, 1].item()
                    decision = self.three_way_decision.get_decisions(
                        torch.tensor([normal_prob])
                    )[0].item()

                    if decision == 0:
                        _, pred_idx = prob.max(1)
                        label = self.class_names[pred_idx.item()]
                    elif decision == 1:
                        label = '异常'
                    else:
                        label = '不确定'

                self.results[label].append(fn)
            except:
                self.results['异常'].append(fn)
        
        auto_count = len(self.results['晨读']) + len(self.results['晨跑'])
        need_review = len(self.results['异常']) + len(self.results['不确定'])
        
        self.info_text.insert(tk.END, f"\n{'='*60}\n")
        self.info_text.insert(tk.END, f"  预测完成!\n")
        self.info_text.insert(tk.END, f"{'='*60}\n")
        self.info_text.insert(tk.END, f"  自动通过: {auto_count} 张 ({auto_count/len(files)*100:.1f}%)\n")
        self.info_text.insert(tk.END, f"  需要人工审核: {need_review} 张 ({need_review/len(files)*100:.1f}%)\n")
        self.info_text.insert(tk.END, f"    - 晨读: {len(self.results['晨读'])} 张\n")
        self.info_text.insert(tk.END, f"    - 晨跑: {len(self.results['晨跑'])} 张\n")
        self.info_text.insert(tk.END, f"    - 异常: {len(self.results['异常'])} 张\n")
        self.info_text.insert(tk.END, f"    - 不确定: {len(self.results['不确定'])} 张\n")
        self.info_text.insert(tk.END, f"{'='*60}\n")
        
        self.review_queue = self.results['异常'] + self.results['不确定']
        self.status_label.config(text=f"预测完成 | 自动通过: {auto_count} | 待审核: {need_review}")
    
    def start_review(self):
        if not self.review_queue:
            messagebox.showinfo("提示", "没有需要审核的图片!")
            return
        
        ReviewWindow(self.current_data_dir, self.review_queue, self.results, self.info_text, self.root)
    
    def generate_report(self):
        if not self.results:
            messagebox.showwarning("提示", "请先运行预测!")
            return
        
        total = sum(len(v) for v in self.results.values())
        
        report = {
            'summary': {
                'total': total,
                '晨读': len(self.results['晨读']),
                '晨跑': len(self.results['晨跑']),
                '异常': len(self.results['异常']),
                '不确定': len(self.results['不确定']),
                'auto_pass': len(self.results['晨读']) + len(self.results['晨跑']),
                'need_review': len(self.results['异常']) + len(self.results['不确定'])
            },
            'abnormal_list': self.results['异常'],
            'uncertain_list': self.results['不确定']
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        report_file = os.path.join(self.output_dir, f'report_{len(os.listdir(self.output_dir))}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.info_text.insert(tk.END, f"\n{'='*60}\n")
        self.info_text.insert(tk.END, f"  报告已保存: {report_file}\n")
        self.info_text.insert(tk.END, f"{'='*60}\n")
        
        self.info_text.insert(tk.END, f"\n  汇总:\n")
        self.info_text.insert(tk.END, f"    总计: {report['summary']['total']} 张\n")
        self.info_text.insert(tk.END, f"    晨读: {report['summary']['晨读']} 张\n")
        self.info_text.insert(tk.END, f"    晨跑: {report['summary']['晨跑']} 张\n")
        self.info_text.insert(tk.END, f"    异常: {report['summary']['异常']} 张\n")
        self.info_text.insert(tk.END, f"    不确定: {report['summary']['不确定']} 张\n")
        
        self.status_label.config(text=f"报告已生成 | {os.path.basename(report_file)}")


class ReviewWindow:
    def __init__(self, data_dir, review_queue, results_dict, info_text, parent_root):
        self.data_dir = data_dir
        self.review_queue = review_queue
        self.original_queue = review_queue.copy()
        self.results = results_dict
        self.info_text = info_text
        self.current_idx = 0
        self.filter_mode = '待审核'
        self.corrections = []
        
        self.win = tk.Toplevel(parent_root)
        self.win.title("✏️ 人工审核")
        self.win.geometry("1050x900")
        self.win.configure(bg='#e8f4f8')
        
        top_frame = tk.Frame(self.win, bg='#1a5f7a', height=60)
        top_frame.pack(fill=tk.X)
        top_frame.pack_propagate(False)
        
        top_label_frame = tk.Frame(top_frame, bg='#1a5f7a')
        top_label_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.label = tk.Label(top_label_frame, text="📋 人工审核窗口", 
                             font=('Microsoft YaHei', 14, 'bold'), bg='#1a5f7a', fg='white')
        self.label.pack(side=tk.LEFT)
        
        self.progress = tk.Label(top_label_frame, text="⏳ 进度: 0/0", 
                                font=('Microsoft YaHei', 12), bg='#1a5f7a', fg='#87ceeb')
        self.progress.pack(side=tk.RIGHT)
        
        filter_frame = tk.Frame(self.win, bg='#d4e6ed', height=50)
        filter_frame.pack(fill=tk.X, padx=20, pady=10)
        filter_frame.pack_propagate(False)
        
        filter_inner = tk.Frame(filter_frame, bg='#d4e6ed')
        filter_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)
        
        tk.Label(filter_inner, text="🔽 筛选:", font=('Microsoft YaHei', 11), 
                bg='#d4e6ed', fg='#1a5f7a').pack(side=tk.LEFT, padx=(0, 5))
        self.filter_combo = ttk.Combobox(filter_inner, 
                                        values=['待审核', '全部', '晨读', '晨跑', '异常', '不确定'], 
                                         state='readonly', width=12, font=('Microsoft YaHei', 11))
        self.filter_combo.set('待审核')
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind('<<ComboboxSelected>>', lambda e: self.on_filter_change())
        
        self.error_label = tk.Label(filter_inner, text="📊 纠正: 0 | 误判率: 0.00%", 
                                    font=('Microsoft YaHei', 11, 'bold'), bg='#d4e6ed', fg='#c0392b')
        self.error_label.pack(side=tk.LEFT, padx=25)
        
        export_btn = tk.Button(filter_inner, text="📥 导出报告", command=self.export_report, 
                              width=12, font=('Microsoft YaHei', 10, 'bold'), 
                              bg='#9b59b6', fg='white', relief=tk.RAISED, cursor='hand2', bd=2)
        export_btn.pack(side=tk.RIGHT, padx=5)
        
        canvas_frame = tk.Frame(self.win, bg='#1a5f7a', padx=3, pady=3)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg='#ffffff', highlightthickness=2,
                               highlightcolor='#3498db', relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = tk.Frame(self.win, bg='#e8f4f8')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        instruction_label = tk.Label(btn_frame, text="⌨️ 快捷键操作", 
                                     font=('Microsoft YaHei', 12, 'bold'), 
                                     bg='#e8f4f8', fg='#1a5f7a')
        instruction_label.pack(pady=(0, 5))
        
        keys_frame = tk.Frame(btn_frame, bg='#ffffff', bd=2, relief=tk.RAISED)
        keys_frame.pack(pady=5, padx=10, fill=tk.X)
        
        key_labels = [
            ("🔵 按 1 = 晨读", '#3498db'),
            ("🟢 按 2 = 晨跑", '#27ae60'),
            ("🔴 按 3 = 异常", '#e74c3c'),
            ("⚪ 按 0 = 跳过", '#95a5a6'),
            ("🟣 ← → = 导航", '#9b59b6')
        ]
        
        for text, color in key_labels:
            tk.Label(keys_frame, text=text, font=('Microsoft YaHei', 10, 'bold'), 
                    bg='#ffffff', fg=color, padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        
        nav_btn = tk.Frame(btn_frame, bg='#e8f4f8')
        nav_btn.pack(pady=8)
        
        prev_btn = tk.Button(nav_btn, text="⬅️ 上一张 (←)", command=self.prev_image, width=12,
                            font=('Microsoft YaHei', 10, 'bold'), bg='#3498db', fg='white',
                            relief=tk.RAISED, cursor='hand2', bd=2)
        prev_btn.pack(side=tk.LEFT, padx=8, ipady=3)
        
        next_btn = tk.Button(nav_btn, text="下一张 (→) ➡️", command=self.next_image, width=12,
                            font=('Microsoft YaHei', 10, 'bold'), bg='#3498db', fg='white',
                            relief=tk.RAISED, cursor='hand2', bd=2)
        next_btn.pack(side=tk.LEFT, padx=8, ipady=3)
        
        self.win.bind('<Key-1>', lambda e: self.set_label('晨读'))
        self.win.bind('<Key-2>', lambda e: self.set_label('晨跑'))
        self.win.bind('<Key-3>', lambda e: self.set_label('异常'))
        self.win.bind('<Key-0>', lambda e: self.skip())
        self.win.bind('<Left>', lambda e: self.prev_image())
        self.win.bind('<Right>', lambda e: self.next_image())
        
        self.show_image()
    
    def on_filter_change(self):
        self.filter_mode = self.filter_combo.get()
        
        if self.filter_mode == '待审核':
            self.review_queue = self.original_queue.copy()
        elif self.filter_mode == '全部':
            all_files = []
            for label in ['晨读', '晨跑', '异常', '不确定']:
                all_files.extend(self.results.get(label, []))
            self.review_queue = all_files
        else:
            self.review_queue = self.results.get(self.filter_mode, []).copy()
        
        self.current_idx = 0
        if self.review_queue:
            self.show_image()
        else:
            messagebox.showinfo("提示", f"没有 {self.filter_mode} 的图片")
    
    def export_report(self):
        corrected = len(self.corrections)
        total_decided = len(self.results.get('晨读', [])) + len(self.results.get('晨跑', []))
        error_rate = corrected / total_decided * 100 if total_decided > 0 else 0
        
        report = {
            'summary': {
                'total_reviewed': len(self.original_queue),
                'corrections': corrected,
                'error_rate': error_rate,
                'results': {
                    '晨读': len(self.results.get('晨读', [])),
                    '晨跑': len(self.results.get('晨跑', [])),
                    '异常': len(self.results.get('异常', [])),
                    '不确定': len(self.results.get('不确定', []))
                }
            },
            'corrections': self.corrections
        }
        
        report_file = os.path.join(os.path.dirname(self.data_dir), 'review_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        messagebox.showinfo("导出成功", f"报告已保存到:\n{report_file}")
    
    def show_image(self):
        if self.current_idx >= len(self.review_queue):
            self.label.config(text="审核完成!")
            return
        
        fn = self.review_queue[self.current_idx]
        self.progress.config(text=f"{self.current_idx+1}/{len(self.review_queue)}")
        
        orig_label = None
        for label, files in self.results.items():
            if fn in files:
                orig_label = label
                break
        
        self.label.config(text=f"{fn}  (原标签: {orig_label})")
        
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
        
        orig_label = None
        for l, files in self.results.items():
            if fn in files:
                orig_label = l
                break
        
        if orig_label and orig_label != label:
            if orig_label in ['晨读', '晨跑'] and label in ['晨读', '晨跑']:
                self.corrections.append({'file': fn, 'from': orig_label, 'to': label})
                self.update_error_rate()
        
        for l in ['晨读', '晨跑', '异常', '不确定']:
            if fn in self.results.get(l, []):
                self.results[l].remove(fn)
        
        self.results[label].append(fn)
        self.current_idx += 1
        self.show_image()
    
    def update_error_rate(self):
        corrected = len(self.corrections)
        total_decided = len(self.results.get('晨读', [])) + len(self.results.get('晨跑', []))
        
        if total_decided > 0:
            error_rate = corrected / total_decided * 100
        else:
            error_rate = 0
        
        self.error_label.config(text=f"纠正: {corrected} | 误判率: {error_rate:.2f}%")
    
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
    CheckInSystem()


if __name__ == '__main__':
    main()