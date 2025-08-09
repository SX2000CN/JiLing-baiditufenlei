import json
from pathlib import Path

# 1. 自动生成 class_names.json，必须在任何读取之前执行
base_dir   = Path(__file__).parent
train_dir  = base_dir / "train"                        # ← 确保这是你的训练集根目录
json_path  = base_dir / "class_names.json"

if not json_path.exists():
    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

with open(json_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)