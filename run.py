"""一键启动入口。

1. 自动检查模型是否已训练；未训练则自动完成"数据生成 + 训练"
2. 启动 Web 服务（http://127.0.0.1:8000）

用法：
    python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _console_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ensure_models() -> None:
    from app.config import MODEL_DIR

    required = ("fault_classifier.joblib", "severity_regressor.joblib",
                "rul_regressor.joblib", "label_encoder.joblib")
    if all((MODEL_DIR / n).exists() for n in required):
        print("[run] 已检测到训练好的模型，跳过训练")
        return
    print("[run] 未检测到模型，开始自动生成数据集并训练（约 1 分钟）...")
    from training.generate_dataset import main as generate
    from training.train_models import main as train

    generate()
    train()


def main() -> None:
    _console_utf8()
    ensure_models()

    import uvicorn

    from app.config import HOST, PORT

    print("=" * 56)
    print("  轴承故障预测性维护系统")
    print(f"  监控看板: http://{HOST}:{PORT}")
    print(f"  API 文档: http://{HOST}:{PORT}/docs")
    print("  按 Ctrl+C 退出")
    print("=" * 56)
    uvicorn.run("app.main:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
