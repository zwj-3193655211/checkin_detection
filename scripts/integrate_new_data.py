"""
将新的23-24晨读晨练签到数据整合到训练集
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def integrate_new_training_data():
    base_dir = Path(r'C:\Users\31936\Desktop\晨读晨练签到打卡检测')
    new_data_dir = Path(r'C:\Users\31936\Downloads\23-24晨读晨练签到')
    data_dir = base_dir / 'checkin_detection' / 'data'
    raw_dir = data_dir / 'raw'
    labels_file = base_dir / 'checkin_detection' / 'data' / 'labels.json'
    review_report_file = base_dir / 'checkin_detection' / 'review_report.json'
    
    if not new_data_dir.exists():
        print(f"❌ 新数据文件夹不存在: {new_data_dir}")
        return
    
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建图片目录: {raw_dir}")
    
    print(f"📂 新数据目录: {new_data_dir}")
    print(f"📂 目标图片目录: {raw_dir}")
    print(f"📄 标签文件: {labels_file}")
    
    jpeg_files = list(new_data_dir.glob('*.jpeg')) + list(new_data_dir.glob('*.jpg'))
    print(f"\n📊 发现 {len(jpeg_files)} 个图片文件")
    
    copied_count = 0
    skipped_count = 0
    already_exists = []
    
    for img_file in jpeg_files:
        target_path = raw_dir / img_file.name
        
        if target_path.exists():
            already_exists.append(img_file.name)
            skipped_count += 1
            continue
        
        try:
            shutil.copy2(img_file, target_path)
            copied_count += 1
            if copied_count % 50 == 0:
                print(f"  已复制: {copied_count}/{len(jpeg_files)}")
        except Exception as e:
            print(f"  ❌ 复制失败 {img_file.name}: {e}")
    
    print(f"\n✅ 复制完成:")
    print(f"  - 新增图片: {copied_count}")
    print(f"  - 跳过(已存在): {skipped_count}")
    
    if review_report_file.exists():
        try:
            with open(review_report_file, 'r', encoding='utf-8') as f:
                review_data = json.load(f)
            
            results = review_data.get('summary', {}).get('results', {})
            corrections = review_data.get('corrections', [])
            
            print(f"\n📋 从审核报告中提取标签:")
            print(f"  - 晨读: {results.get('晨读', 0)} 张")
            print(f"  - 晨跑: {results.get('晨跑', 0)} 张")
            print(f"  - 异常: {results.get('异常', 0)} 张")
            print(f"  - 纠正记录: {len(corrections)} 条")
            
            if labels_file.exists():
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
            else:
                labels_data = {"_schema": "checkin_labels_v1", "labels": {}}
            
            new_labels = {}
            for filename in jpeg_files:
                filename_str = filename.name
                
                correction = next((c for c in corrections if c['file'] == filename_str), None)
                
                if filename_str.startswith(('1-', '2-', '3-', '4-', '5-', '6-', '7-', '8-', '9-')):
                    new_labels[filename_str] = {
                        "label": "晨读",
                        "scene": "morning_reading",
                        "is_normal": "normal",
                        "timestamp": f"2026-{filename_str.split('-')[1]}-{filename_str.split('-')[2].split('.')[0]} 08:00:00",
                        "source": "23-24晨读晨练签到"
                    }
            
            labels_data['labels'].update(new_labels)
            
            with open(labels_file, 'w', encoding='utf-8') as f:
                json.dump(labels_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 标签文件已更新:")
            print(f"  - 新增标签: {len(new_labels)} 条")
            print(f"  - 总计标签: {len(labels_data['labels'])} 条")
            
        except Exception as e:
            print(f"\n⚠️ 处理标签时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️ 未找到审核报告文件: {review_report_file}")
        print("  将复制所有图片但不会更新标签")
    
    print(f"\n✨ 整合完成!")
    print(f"\n📝 下一步:")
    print(f"  1. 检查 data/raw 目录中的新图片")
    print(f"  2. 运行训练脚本更新模型: python src/train_resnet.py")
    print(f"  3. 运行预处理脚本分析数据: python scripts/preprocessing.py")

if __name__ == '__main__':
    print("=" * 60)
    print("晨读晨练数据整合工具")
    print("=" * 60)
    integrate_new_training_data()
