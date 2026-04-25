import json
import os
from pathlib import Path

raw_dir = Path('data/raw')
labels_file = Path('data/labels.json')

jpeg_files = list(raw_dir.glob('*.jpeg')) + list(raw_dir.glob('*.jpg'))
print(f"📂 data/raw 目录图片数: {len(jpeg_files)}")

if labels_file.exists():
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    label_count = len(labels_data.get('labels', {}))
    print(f"📄 标签文件中的标签数: {label_count}")
    
    if label_count < len(jpeg_files):
        print(f"⚠️ 缺少标注的图片数: {len(jpeg_files) - label_count}")
    else:
        print(f"✅ 所有图片都有标注")
    
    label_stats = {'晨读': 0, '晨跑': 0, '异常': 0, '不确定': 0}
    for label_info in labels_data.get('labels', {}).values():
        label_name = label_info.get('label', '未知')
        if label_name in label_stats:
            label_stats[label_name] += 1
    
    print(f"\n📊 标签统计:")
    for label, count in label_stats.items():
        if count > 0:
            print(f"  - {label}: {count} 张")
else:
    print(f"❌ 标签文件不存在")
