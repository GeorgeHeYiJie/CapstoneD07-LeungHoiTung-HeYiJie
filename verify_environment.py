"""Import every repository Python file without invoking its command-line workflow."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = {
    "torch": "torch", "numpy": "numpy", "pandas": "pandas",
    "scikit-learn": "sklearn", "matplotlib": "matplotlib",
    "SciencePlots": "scienceplots", "seaborn": "seaborn",
    "openpyxl": "openpyxl", "PyAutoGUI": "pyautogui", "keyboard": "keyboard",
}


def import_file(relative_path):
    path = ROOT / relative_path
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(path)]
    name = "_import_check_" + hashlib.sha256(relative_path.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--report", default="environment_checks/import_report.json")
    args = parser.parse_args()
    if args.child:
        import_file(args.child)
        return 0

    output = ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output.parent / "matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"

    dependencies = []
    for distribution, module_name in DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
            dependencies.append({"package": distribution,
                                 "version": importlib.metadata.version(distribution),
                                 "status": "PASS"})
        except Exception as exc:
            dependencies.append({"package": distribution, "status": "FAIL",
                                 "error": repr(exc)})

    paths = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
                   if not any(part.startswith(".") or part in
                              {"__pycache__", "environment_checks"}
                              for part in p.relative_to(ROOT).parts))

    def check(relative_path):
        try:
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__)), "--child", relative_path],
                cwd=ROOT, env=os.environ.copy(), capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            return {"file": relative_path,
                    "status": "PASS" if result.returncode == 0 else "FAIL",
                    "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"file": relative_path, "status": "FAIL", "error": "60 s timeout"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        modules = list(pool.map(check, paths))

    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                               capture_output=True, text=True, timeout=60)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "dependencies": dependencies, "modules": modules,
        "pip_check": {"returncode": pip_check.returncode,
                      "stdout": pip_check.stdout, "stderr": pip_check.stderr},
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["Model/Model.py", "Model/Compare_Models.py",
                         "dataloader/dataloader.py", "utils/util.py"]
        },
        "scope": "Actual imports only; no training, plotting or dataset evaluation.",
    }
    if any(d["package"] == "torch" and d["status"] == "PASS" for d in dependencies):
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in dependencies + modules:
        print(item["status"], item.get("file", item.get("package")))
        if item["status"] == "FAIL":
            print(item.get("stderr", item.get("error", "")))
    print(pip_check.stdout.strip())
    print(f"Report: {output}")
    return int(pip_check.returncode != 0 or any(
        item["status"] != "PASS" for item in dependencies + modules))


if __name__ == "__main__":
    raise SystemExit(main())
