"""
更新《实训项目实践报告.docx》：
1. 数据集 2057→2067、训练/验证/测试划分数字
2. 3.4/5.1/5.5 测试集指标（审核率23.5%→25.9%、通过率76.5%→74.1%）
3. 修正阈值描述（≥0.85 → ≥0.80，与 config.py 的 ALPHA_AUTO_PASS=0.80 一致）
4. 5.4.2 特征识别优化补充 11 独立 MLP 实验结论
5. 图编号重复修复、参考文献[1]模板残留替换
"""
import docx
from docx import Document

DOC = '实训项目实践报告.docx'


def replace_in_runs(par, old, new):
    """在段落所有run中做文本替换（保留run结构，不影响inline图片）"""
    full = ''.join(r.text for r in par.runs)
    if old not in full:
        return False
    # 简单情况：old 在单个 run 内
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    # 跨 run：重建文本到第一个含内容的 run（用索引定位，避免身份比较陷阱）
    runs = list(par.runs)
    first_idx = 0
    for i, r in enumerate(runs):
        if r.text.strip():
            first_idx = i
            break
    new_full = full.replace(old, new)
    runs[first_idx].text = new_full
    for r in runs[:first_idx] + runs[first_idx + 1:]:
        r.text = ''
    return True


def replace_par_exact(par, old, new):
    """整段精确替换（old为整段文本）"""
    full = ''.join(r.text for r in par.runs)
    if full.strip() != old.strip():
        return False
    runs = list(par.runs)
    runs[0].text = new
    for r in runs[1:]:
        r.text = ''
    return True


def main():
    d = Document(DOC)
    stats = {'para': 0, 'table': 0, 'ref': 0}

    # ---------- 段落更新 ----------
    for p in d.paragraphs:
        full = ''.join(r.text for r in p.runs)

        # 阈值描述修正（规则1/2）
        if '高置信度（≥0.85）或中高置信度（≥0.8）+匹配特征≥5' in full:
            replace_in_runs(p, '：高置信度（≥0.85）或中高置信度（≥0.8）+匹配特征≥5', '：置信度≥0.80 且匹配特征≥5')
            stats['para'] += 1
        elif '高置信度（≥0.85）或中高置信度（≥0.8）+匹配特征≥3' in full:
            replace_in_runs(p, '：高置信度（≥0.85）或中高置信度（≥0.8）+匹配特征≥3', '：置信度≥0.80 且匹配特征≥3')
            stats['para'] += 1
        # 训练样本数
        elif '数据集规模适中（1422训练样本）' in full:
            replace_in_runs(p, '1422', '1429'); stats['para'] += 1
        elif '数据量充足（1422张训练样本）' in full:
            replace_in_runs(p, '1422', '1429'); stats['para'] += 1
        # 测试集数量
        elif '测试集319进行端到端测试' in full:
            replace_in_runs(p, '319', '321'); stats['para'] += 1
        # 消融实验数据集
        elif '消融实验使用全部数据集（训练集+验证集+测试集，2057张）' in full:
            replace_in_runs(p, '2057', '2067'); stats['para'] += 1
        elif '注：消融实验使用全数据集（2057张）' in full:
            replace_in_runs(p, '2057', '2067'); stats['para'] += 1
        # 性能指标（5.1 / 5.5）
        elif '实现100%准确率、0%漏检率、23.5%人工审核率、76.5%自动通过率' in full:
            replace_in_runs(p, '23.5%人工审核率、76.5%自动通过率', '25.9%人工审核率、74.1%自动通过率')
            stats['para'] += 1
        elif '自动通过率达76.5%' in full:
            replace_in_runs(p, '76.5%', '74.1%'); stats['para'] += 1
        elif '将人工审核率控制在23.5%' in full:
            replace_in_runs(p, '23.5%', '25.9%'); stats['para'] += 1
        # 特征识别优化段落（5.4.2）
        elif '将原有的11维输出mlp模型拆成多个mlp模型' in full:
            new_text = ('特征识别优化：曾尝试将11维输出MLP拆分为11个独立MLP分别识别单个特征'
                        '（每特征独立Focal Loss权重与温度），实验表明：特征准确率与共享结构持平，'
                        '但自动通过样本中依赖错误特征预测的比例从0.5%升至4.5%，可解释性与可靠性下降，'
                        '故维持共享网络结构。后续可通过收集更多标注样本、完善特征标注来提升识别精度')
            replace_par_exact(p, full, new_text)
            stats['para'] += 1
        # 图编号重复（图3 → 图4，第三个图注）
        elif full.strip() == '图3 待审核图片展示3':
            replace_in_runs(p, '图3', '图4'); stats['para'] += 1

    # ---------- 表格更新 ----------
    for t in d.tables:
        for r in t.rows:
            cells = [c.text.strip() for c in r.cells]
            joined = '|'.join(cells)
            # 数据集划分表
            if '训练集' in cells[0] and '1422' in joined:
                r.cells[1].text = '1429张'
                r.cells[2].text = '69.1%'
                stats['table'] += 1
            elif '验证集' in cells[0] and '316' in joined:
                r.cells[1].text = '317张'
                r.cells[2].text = '15.3%'
                stats['table'] += 1
            elif '测试集' in cells[0] and '319' in joined:
                r.cells[1].text = '321张'
                r.cells[2].text = '15.5%'
                stats['table'] += 1
            elif '总计' in cells[0] and '2057' in joined:
                r.cells[1].text = '2067张'
                r.cells[2].text = '100%'
                stats['table'] += 1
            # 类别分布表
            elif cells[0] == '晨读' and '1497' in joined:
                r.cells[2].text = '72.4%'
                stats['table'] += 1
            elif cells[0] == '晨跑' and '536' in joined:
                r.cells[1].text = '546张'
                r.cells[2].text = '26.4%'
                stats['table'] += 1
            # 性能指标表（目标值/实际值）
            elif cells[0] == '人工审核率' and '23.51' in joined:
                r.cells[2].text = '25.86%'
                stats['table'] += 1
            elif cells[0] == '自动通过率' and '76.49' in joined:
                r.cells[2].text = '74.14%'
                stats['table'] += 1
            # 决策结果表
            elif cells[0] == '自动通过' and '244' in joined:
                r.cells[1].text = '238'
                r.cells[2].text = '74.14%'
                stats['table'] += 1
            elif cells[0] == '待审核' and '75' in joined and '23.51' in joined:
                r.cells[1].text = '83'
                r.cells[2].text = '25.86%'
                stats['table'] += 1
            # 规则触发表
            elif '规则1（晨跑自动通过）' in cells[0] and '24' in joined:
                r.cells[1].text = '23'; r.cells[2].text = '7.2%'
                stats['table'] += 1
            elif '规则2（晨读自动通过）' in cells[0] and '186' in joined:
                r.cells[1].text = '178'; r.cells[2].text = '55.5%'
                stats['table'] += 1
            elif '规则3（特征少待审核）' in cells[0] and '74' in joined:
                r.cells[1].text = '76'; r.cells[2].text = '23.7%'
                stats['table'] += 1
            elif '规则4（置信度低待审核）' in cells[0] and '1' in joined:
                r.cells[1].text = '7'; r.cells[2].text = '2.2%'
                stats['table'] += 1
            elif '自动通过(默认)' in cells[0] and '244' in joined:
                r.cells[1].text = '238'; r.cells[2].text = '74.1%'
                stats['table'] += 1

    # ---------- 参考文献[1]模板残留替换 ----------
    for p in d.paragraphs:
        full = ''.join(r.text for r in p.runs)
        if '样例' in full and '网络空间威胁情报' in full:
            new_ref = ('[1] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object '
                       'Detection[C]//Proceedings of the IEEE International Conference on '
                       'Computer Vision. 2017: 2980-2988.')
            replace_par_exact(p, full, new_ref)
            stats['ref'] += 1

    d.save(DOC)
    print(f"更新完成: 段落 {stats['para']} 处, 表格 {stats['table']} 处, 参考文献 {stats['ref']} 处")


if __name__ == '__main__':
    main()
