"""检查「GPU训练要用的」，并在确认后启动高维矩阵模型训练。

训练脚本是仓库里 HighDimensionMatrix-PINN_Combination.py 的原样副本，这里不改它的公式。
数据放在本目录的 data/ 下，因为那个脚本用自己所在目录寻找：

- data/MIT data/<日期批次>/*.csv
- data/HUST data/*.csv

默认只检查文件。加上 --run 和 --yes 才会训练，并且要求当前 PyTorch 能看到 CUDA。
"""

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SCRIPT = ROOT / "HighDimensionMatrix-PINN_Combination.py"
SOURCE = REPO / "HighDimensionMatrix-PINN_Combination.py"
MIT = ROOT / "data" / "MIT data"
HUST = ROOT / "data" / "HUST data"
WORK = ROOT / "运行输出"

MIT_BATCHES = {
    "2017-05-12": 46,
    "2017-06-30": 43,
    "2018-04-12": 36,
}

RUNS = {
    "MIT": {
        "held_out": "2018-04-12",
        "note": "留出 MIT 的 2018-04-12 整批做测试，其余电池按 random_state=420 划 80/20 训练和验证。",
    },
    "HUST": {
        "held_out": "10",
        "note": "留出 HUST 文件名组号 10 做测试，其余电池按 random_state=420 划 80/20 训练和验证。",
    },
}


def csv_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.csv") if path.is_file())


def check_files() -> list[str]:
    problems: list[str] = []
    if not SCRIPT.is_file():
        problems.append("缺少 HighDimensionMatrix-PINN_Combination.py")
    else:
        py_compile.compile(str(SCRIPT), doraise=True)
        if SOURCE.is_file() and SCRIPT.read_bytes() != SOURCE.read_bytes():
            problems.append("训练脚本和仓库根目录的原文件不一致")

    mit_files = csv_files(MIT)
    if len(mit_files) != 125:
        problems.append(f"MIT 应有 125 个 csv，实际 {len(mit_files)}")
    for batch, count in MIT_BATCHES.items():
        found = list((MIT / batch).glob("*.csv")) if (MIT / batch).is_dir() else []
        if len(found) != count:
            problems.append(f"MIT {batch} 应有 {count} 个 csv，实际 {len(found)}")

    hust_files = list(HUST.glob("*.csv")) if HUST.is_dir() else []
    nested = [path for path in csv_files(HUST) if path.parent != HUST]
    if len(hust_files) != 77 or nested:
        problems.append(f"HUST 应有 77 个 csv 直接放在 data/HUST data，实际顶层 {len(hust_files)} 个，子目录 {len(nested)} 个")

    if SOURCE.is_file():
        for copy in mit_files + hust_files:
            relative = copy.relative_to(ROOT / "data")
            original = REPO / "data" / relative
            if not original.is_file() or copy.read_bytes() != original.read_bytes():
                problems.append(f"数据和仓库原文件不一致: {relative}")
                break
    return problems


def training_command(name: str, save_dir: Path) -> list[str]:
    spec = RUNS[name]
    return [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        name,
        "--held-out",
        spec["held_out"],
        "--epochs",
        "200",
        "--early-stop",
        "20",
        "--max-train",
        "0",
        "--max-val",
        "0",
        "--max-test",
        "0",
        "--stride",
        "1",
        "--batch-size",
        "512",
        "--seed",
        "420",
        "--device",
        "cuda",
        "--save-dir",
        str(save_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="检查或启动高维矩阵 GPU 训练")
    parser.add_argument("--check", action="store_true", help="只检查脚本、数据和语法")
    parser.add_argument("--run", choices=sorted(RUNS), help="训练 MIT 或 HUST")
    parser.add_argument("--yes", action="store_true", help="确认开始训练。不加这个参数不会启动训练")
    args = parser.parse_args()

    problems = check_files()
    if problems:
        print("检查没有通过:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("文件检查通过: 训练脚本 1 个，MIT csv 125 个，HUST csv 77 个。")
    print("脚本是仓库原文件的副本，公式没有改。")
    try:
        import torch
    except Exception as exc:
        print(f"当前 Python 还不能导入 torch（{exc}）。")
        print("请先按 requirements-gpu.txt 安装 CUDA 版 PyTorch。")
        if args.run:
            print("这次没有开始训练。")
            return 1 if args.yes else 0
        print("文件已经齐了。装好 CUDA 版 torch 后再运行一次 --check，确认这台机器能看到 GPU。")
        return 0

    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "无"
    print(f"torch {torch.__version__}，CUDA 可用: {cuda}，设备: {device_name}")

    if not args.run:
        print("这次只做了检查，没有训练。")
        print("开跑示例: python 开始训练.py --run MIT --yes")
        return 0

    print(RUNS[args.run]["note"])
    print("最多 200 个 epoch。验证集连续 21 个 epoch 没有更低的 Wang MSE 会提前停止。")
    print("SOH 是容量 / 1.1 Ah 的小数。checkpoint 保存验证集 Wang MSE 最低的那一轮，Zhang 用同一轮权重。")
    if not args.yes:
        print("这次没有开始训练。确认后加上 --yes。")
        return 0
    if not cuda:
        print("当前 PyTorch 看不到 CUDA，本次没有开始训练，避免在 CPU 上跑完整实验。")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = WORK / f"{args.run}_{RUNS[args.run]['held_out']}_{stamp}"
    save_dir.mkdir(parents=True, exist_ok=False)
    command = training_command(args.run, save_dir)
    print("工作目录里的结果不会提交到 Git。")
    print("命令:")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
