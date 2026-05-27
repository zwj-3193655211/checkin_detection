import json

# 读取数据集划分配置
with open('split_config.json', 'r', encoding='utf-8') as f:
    split_data = json.load(f)

# 读取标签数据
with open('labels.json', 'r', encoding='utf-8') as f:
    labels_data = json.load(f)

labels = labels_data['labels']

# 定义数据集划分
datasets = ['train_files', 'val_files', 'test_files']

# 统计每个数据集的类别分布
results = {}

for dataset in datasets:
    if dataset not in split_data:
        continue
    
    files = split_data[dataset]
    counts = {'晨读': 0, '晨跑': 0, '异常': 0, '未知': 0}
    total = len(files)
    
    for filename in files:
        if filename in labels:
            label = labels[filename]['label']
            if label in counts:
                counts[label] += 1
            else:
                counts['未知'] += 1
        else:
            counts['未知'] += 1
    
    results[dataset] = {
        '总数': total,
        '晨读': counts['晨读'],
        '晨跑': counts['晨跑'],
        '异常': counts['异常'],
        '未知': counts['未知']
    }

# 输出统计结果
print("=" * 60)
print("数据集划分统计报告")
print("=" * 60)

for dataset, stats in results.items():
    print(f"\n【{dataset.replace('_files', '')}】")
    print(f"  总数: {stats['总数']} 张")
    print(f"  ├─ 晨读: {stats['晨读']} 张 ({stats['晨读']/stats['总数']*100:.1f}%)")
    print(f"  ├─ 晨跑: {stats['晨跑']} 张 ({stats['晨跑']/stats['总数']*100:.1f}%)")
    print(f"  ├─ 异常: {stats['异常']} 张 ({stats['异常']/stats['总数']*100:.1f}%)")
    print(f"  └─ 未知: {stats['未知']} 张")

# 计算总体统计
total_all = sum(results[d]['总数'] for d in results)
morning_read_all = sum(results[d]['晨读'] for d in results)
morning_run_all = sum(results[d]['晨跑'] for d in results)
abnormal_all = sum(results[d]['异常'] for d in results)

print("\n" + "=" * 60)
print("【总体统计】")
print(f"  数据集总规模: {total_all} 张")
print(f"  ├─ 晨读: {morning_read_all} 张 ({morning_read_all/total_all*100:.1f}%)")
print(f"  ├─ 晨跑: {morning_run_all} 张 ({morning_run_all/total_all*100:.1f}%)")
print(f"  └─ 异常: {abnormal_all} 张 ({abnormal_all/total_all*100:.1f}%)")
print("=" * 60)