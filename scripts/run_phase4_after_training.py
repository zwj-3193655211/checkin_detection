# -*- coding: utf-8 -*-
"""Phase 4 触发器（后台守护）：等三 seed 训练结果齐全后自动跑公平对比+报告

背景：
  - 真实训练进程（pSeIJ8）已脱离任务追踪器但仍在跑（Windows 分离进程），
    resume 代码已合入，正常会产出 seed42/123/777 的 preds+results。
  - 本脚本受追踪地跑在后台：轮询等待 data/vit_tiny_results_seed777.json 出现，
    出现即调用 compare_all_encoders.main() 产出最终报告。
  - 若超过超时（默认 150 分钟）仍未齐全，判定训练疑似中断，以 resume 模式
    重跑缺失 seed，再跑对比。resume-skip 保证已完成的 seed 不重训。

用法（后台）：
    python scripts/run_phase4_after_training.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [42, 123, 777]
TIMEOUT_SEC = 150 * 60


def all_done():
    return all((ROOT / "data" / f"vit_tiny_results_seed{s}.json").exists() for s in SEEDS)


def wait_for_training():
    """等待训练产出齐全；超时则 resume 重跑缺失 seed。返回是否齐全。"""
    deadline = time.time() + TIMEOUT_SEC
    while not all_done():
        if time.time() > deadline:
            print(f"[watch] 超时 {TIMEOUT_SEC//60}min，判定训练中断，resume 重跑缺失 seed…")
            import subprocess
            # 用当前 venv 解释器（sys.executable）重跑，resume-skip 只做缺失 seed
            subprocess.run([sys.executable, str(ROOT / "scripts" / "train_vit_tiny.py"),
                            "--all-seeds"], check=False)
            break
        time.sleep(30)
    return all_done()


def main():
    print(f"[watch] 等待 {SEEDS} 三 seed 训练结果齐全（超时 {TIMEOUT_SEC//60}min）…")
    ready = wait_for_training()
    if not ready:
        print("[watch] resume 重跑后仍未齐全，退出。")
        return
    print("[watch] 三 seed 齐全，开始 Phase 4 公平对比 + 报告…")
    import compare_all_encoders
    compare_all_encoders.main()
    print("\n[watch] Phase 4 完成：EXPERIMENT_REPORT_NAFLEX.md 已写出。")


if __name__ == "__main__":
    main()
