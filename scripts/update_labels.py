"""
为新复制的数据添加标签 - 使用已有的审核结果
"""
import os
import json
from pathlib import Path
from datetime import datetime

def update_labels_for_new_data():
    base_dir = Path(r'C:\Users\31936\Desktop\晨读晨练签到打卡检测')
    raw_dir = base_dir / 'checkin_detection' / 'data' / 'raw'
    labels_file = base_dir / 'checkin_detection' / 'data' / 'labels.json'
    review_report_file = base_dir / 'checkin_detection' / 'review_report.json'
    
    print("=" * 60)
    print("为新数据添加标签")
    print("=" * 60)
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    existing_labels = labels_data['labels']
    print(f"📊 当前标签数量: {len(existing_labels)}")
    
    review_data = None
    if review_report_file.exists():
        with open(review_report_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
    
    corrections = {}
    if review_data:
        for corr in review_data.get('corrections', []):
            corrections[corr['file']] = corr['to']
    
    results = review_data.get('summary', {}).get('results', {})
    
    print(f"\n📋 审核结果统计:")
    print(f"  - 晨读: {results.get('晨读', 0)} 张")
    print(f"  - 晨跑: {results.get('晨跑', 0)} 张")
    print(f"  - 异常: {results.get('异常', 0)} 张")
    print(f"  - 不确定: {results.get('不确定', 0)} 张")
    print(f"  - 纠正记录: {len(corrections)} 条")
    
    jpeg_files = list(raw_dir.glob('*.jpeg')) + list(raw_dir.glob('*.jpg'))
    
    new_files = []
    for img_file in jpeg_files:
        if img_file.name not in existing_labels:
            new_files.append(img_file.name)
    
    print(f"\n📂 新增图片文件: {len(new_files)} 个")
    
    added_count = 0
    skipped_count = 0
    
    for filename in new_files:
        if filename in corrections:
            label = corrections[filename]
            scene = 'morning_reading' if label == '晨读' else 'morning_running'
            
            date_part = filename.replace('.jpeg', '').replace('.jpg', '')
            parts = date_part.split('-')
            if len(parts) >= 3:
                date_str = f"{parts[1]}-{parts[2]}"
            else:
                date_str = "2026-04-23"
            
            existing_labels[filename] = {
                "label": label,
                "scene": scene,
                "is_normal": "normal",
                "timestamp": f"2026-{date_str} 08:00:00",
                "source": "23-24晨读晨练签到",
                "verified": True
            }
            added_count += 1
        else:
            parts = filename.replace('.jpeg', '').replace('.jpg', '').split('-')
            
            if len(parts) >= 3:
                try:
                    month = int(parts[1])
                    day = int(parts[2])
                    
                    if month == 4 and day in [23, 24]:
                        scene = 'morning_reading'
                        label = '晨读'
                    else:
                        continue
                    
                    date_str = f"{parts[1]}-{parts[2]}"
                    
                    existing_labels[filename] = {
                        "label": label,
                        "scene": scene,
                        "is_normal": "normal",
                        "timestamp": f"2026-{date_str} 08:00:00",
                        "source": "23-24晨读晨练签到",
                        "verified": False
                    }
                    added_count += 1
                except (ValueError, IndexError):
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
    
    print(f"\n✅ 标签添加完成:")
    print(f"  - 新增标签: {added_count}")
    print(f"  - 跳过: {skipped_count}")
    print(f"  - 总计标签: {len(existing_labels)}")
    
    labels_data['labels'] = existing_labels
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 标签文件已保存: {labels_file}")
    
    label_stats = {'晨读': 0, '晨跑': 0, '异常': 0}
    for label_info in existing_labels.values():
        label_name = label_info.get('label', '未知')
        if label_name in label_stats:
            label_stats[label_name] += 1
    
    print(f"\n📊 标签统计:")
    for label, count in label_stats.items():
        print(f"  - {label}: {count} 张")
    
    print(f"\n✨ 完成!")

if __name__ == '__main__':
    update_labels_for_new_data()
