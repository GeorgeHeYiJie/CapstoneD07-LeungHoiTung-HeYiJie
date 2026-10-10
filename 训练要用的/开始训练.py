"""把「训练要用的」里的分层目录接成作者脚本能直接训练的样子。

作者的入口脚本写死了这些相对路径：

- ``from Model.Model import PINN``
- ``from dataloader.dataloader import ...``
- ``from utils.util import ...``
- ``data/XJTU data``、``data/HUST data``、``data/MIT data``、``data/TJU data``
- ``./pretrained model/model_*.pth``（只有微调脚本会读）

文件按用途分开放。真正开始训练之前，这个脚本会在「运行输出」里接上数据和权重，
再把模型、数据读取和工具目录放进 Python 路径。作者的 .py 内容不改。

默认只检查文件齐不齐，不会开始训练。
"""

from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORK = ROOT / "运行输出"

PACKAGE_DIRS = [
    ROOT / "02_模型",
    ROOT / "03_读电池数据",
    ROOT / "04_训练工具",
    ROOT / "05_训练和运行脚本" / "四个数据集从零训练",
    ROOT / "05_训练和运行脚本" / "对比基线训练",
    ROOT / "05_训练和运行脚本" / "用已有权重微调",
]

DATA_DIR = ROOT / "06_四个数据集" / "data"
WEIGHTS_DIR = ROOT / "07_直接运行和微调用的权重" / "pretrained model"

SCRIPTS = {
    "XJTU": ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_XJTU.py",
    "HUST": ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_HUST.py",
    "MIT": ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_MIT.py",
    "TJU": ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_TJU.py",
    "comparison": ROOT / "05_训练和运行脚本" / "对比基线训练" / "main_comparision.py",
    "finetune": ROOT / "05_训练和运行脚本" / "用已有权重微调" / "main_adaptation - fine-tuning.py",
}

# 复制来源。检查时如果仓库根目录还在，就逐个比对，确认这份副本没有被改过。
SOURCE_OF = {
    ROOT / "01_环境依赖" / "requirements-reproduction.txt": REPO / "requirements-reproduction.txt",
    ROOT / "02_模型" / "Model" / "__init__.py": REPO / "Model" / "__init__.py",
    ROOT / "02_模型" / "Model" / "Model.py": REPO / "Model" / "Model.py",
    ROOT / "02_模型" / "Model" / "Compare_Models.py": REPO / "Model" / "Compare_Models.py",
    ROOT / "03_读电池数据" / "dataloader" / "__init__.py": REPO / "dataloader" / "__init__.py",
    ROOT / "03_读电池数据" / "dataloader" / "dataloader.py": REPO / "dataloader" / "dataloader.py",
    ROOT / "04_训练工具" / "utils" / "__init__.py": REPO / "utils" / "__init__.py",
    ROOT / "04_训练工具" / "utils" / "util.py": REPO / "utils" / "util.py",
    ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_XJTU.py": REPO / "main_XJTU.py",
    ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_HUST.py": REPO / "main_HUST.py",
    ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_MIT.py": REPO / "main_MIT.py",
    ROOT / "05_训练和运行脚本" / "四个数据集从零训练" / "main_TJU.py": REPO / "main_TJU.py",
    ROOT / "05_训练和运行脚本" / "对比基线训练" / "main_comparision.py": REPO / "main_comparision.py",
    ROOT / "05_训练和运行脚本" / "用已有权重微调" / "main_adaptation - fine-tuning.py": REPO / "main_adaptation - fine-tuning.py",
}

EXPECTED_CSV = {
    "XJTU data": 55,
    "HUST data": 77,
    "MIT data/2017-05-12": 46,
    "MIT data/2017-06-30": 43,
    "MIT data/2018-04-12": 36,
    "TJU data/Dataset_1_NCA_battery": 66,
    "TJU data/Dataset_2_NCM_battery": 55,
    "TJU data/Dataset_3_NCM_NCA_battery": 9,
}

EXPECTED_WEIGHTS = [
    "model_XJTU_0.pth",
    "model_XJTU_1.pth",
    "model_XJTU_2.pth",
    "model_XJTU_3.pth",
    "model_XJTU_4.pth",
    "model_XJTU_5.pth",
    "model_HUST.pth",
    "model_MIT.pth",
    "model_TJU_0.pth",
    "model_TJU_1.pth",
    "model_TJU_2.pth",
]

RUN_NOTE = {
    "XJTU": "会按作者 main() 连续训练 6 个工况 × 10 次实验。每次最多 200 个 epoch。",
    "HUST": "会按作者 main() 连续训练 10 次实验。每次最多 200 个 epoch。",
    "MIT": "会按作者 main() 连续训练 10 次实验。每次最多 200 个 epoch。",
    "TJU": "作者把文件末尾写成了 pass，直接运行不会开始训练。要训练需要在脚本里调用 main()。",
    "comparison": "会按作者文件末尾的循环，用 MLP 在 XJTU 的 6 个工况上各训练 10 次。",
    "finetune": "作者把文件末尾写成了 pass，直接运行不会开始微调。权重已放在第 7 层，供脚本里的函数读取。",
}


def same_bytes(left: Path, right: Path) -> bool:
    return left.read_bytes() == right.read_bytes()


def check_files() -> list[str]:
    problems = []
    for copy_path, source in SOURCE_OF.items():
        if not copy_path.is_file():
            problems.append(f"缺少 {copy_path.relative_to(ROOT)}")
            continue
        if copy_path.suffix == ".py":
            py_compile.compile(str(copy_path), doraise=True)
        if source.is_file() and not same_bytes(copy_path, source):
            problems.append(f"和仓库原文件不一致: {copy_path.relative_to(ROOT)}")

    for folder, count in EXPECTED_CSV.items():
        directory = DATA_DIR / folder
        found = list(directory.glob("*.csv")) if directory.is_dir() else []
        if len(found) != count:
            problems.append(f"{folder} 应有 {count} 个 csv，实际 {len(found)}")
        elif (REPO / "data" / folder).is_dir():
            for csv_path in found:
                source = REPO / "data" / folder / csv_path.name
                if not source.is_file() or not same_bytes(csv_path, source):
                    problems.append(f"数据和原文件不一致: {csv_path.relative_to(ROOT)}")
                    break

    for name in EXPECTED_WEIGHTS:
        weight = WEIGHTS_DIR / name
        source = REPO / "pretrained model" / name
        if not weight.is_file():
            problems.append(f"缺少权重 {name}")
        elif source.is_file() and not same_bytes(weight, source):
            problems.append(f"权重和原文件不一致: {name}")
    return problems


def ensure_directory_link(source: Path, destination: Path) -> str:
    """让工作目录里出现作者脚本要找的文件夹名。优先用链接，避免再复制一份数据。"""
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return "已有链接"
        destination.unlink()
    elif destination.exists():
        return "已有目录"
    try:
        os.symlink(source, destination, target_is_directory=True)
        return "新建链接"
    except OSError:
        shutil.copytree(source, destination)
        return "无法建立链接，已复制一份"


def prepare_workdir() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    data_status = ensure_directory_link(DATA_DIR, WORK / "data")
    weight_status = ensure_directory_link(WEIGHTS_DIR, WORK / "pretrained model")
    print(f"工作目录: {WORK}")
    print(f"data: {data_status}")
    print(f"pretrained model: {weight_status}")
    return WORK


def run_script(name: str) -> None:
    script = SCRIPTS[name]
    for package_dir in PACKAGE_DIRS:
        sys.path.insert(0, str(package_dir))
    os.chdir(prepare_workdir())
    # 作者脚本会解析 sys.argv。这里只留下脚本名，避免把 --run 传进去。
    sys.argv = [str(script)]
    print(f"开始运行: {script.relative_to(ROOT)}")
    print(RUN_NOTE[name])
    print("日志和 model.pth 会写到上面的工作目录里，相对路径跟作者脚本原来的设置一样。")
    import runpy
    runpy.run_path(str(script), run_name="__main__")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查或启动「训练要用的」里的作者训练脚本")
    parser.add_argument("--check", action="store_true", help="只检查文件、语法和是否与仓库原文件一致")
    parser.add_argument("--run", choices=sorted(SCRIPTS), help="运行对应的作者入口脚本")
    parser.add_argument("--yes", action="store_true", help="确认开始训练。不加这个参数不会启动训练")
    args = parser.parse_args()

    problems = check_files()
    if problems:
        print("检查没有通过:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    csv_count = sum(1 for _ in DATA_DIR.rglob("*.csv"))
    print(f"文件检查通过: 训练代码 {len(SOURCE_OF)} 个，电池 csv {csv_count} 个，预训练权重 {len(EXPECTED_WEIGHTS)} 个。")
    try:
        import torch
        print(f"当前 Python 可以导入 torch {torch.__version__}。")
    except Exception as exc:
        print(f"当前 Python 还不能导入 torch（{exc}）。文件已经齐了，训练前请先安装 01_环境依赖/requirements-reproduction.txt。")

    if args.run:
        if not args.yes:
            print("这次没有开始训练。确认要跑完整作者流程时，加上 --yes。")
            print(RUN_NOTE[args.run])
            return 0
        run_script(args.run)
        return 0

    print("这次只做了检查，没有训练。要开跑请使用: python 开始训练.py --run XJTU --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
