"""FastAPI 应用入口：装配数据库、诊断模型与监测引擎，挂载前端看板。

启动方式：
    python run.py           # 一键启动（自动检查/训练模型）
    python -m app.main      # 直接启动
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import HOST, PORT, STATIC_DIR
from app.database import Database
from app.routers import diagnose, history, monitor
from app.services.fault_detector import DiagnosticModels
from app.services.monitor import MonitorEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 装配全局单例（数据库 / 模型 / 监测引擎）
    db = Database()
    models = DiagnosticModels()
    engine = MonitorEngine(db, models)
    app.state.db = db
    app.state.models = models
    app.state.engine = engine
    # 预热加载模型，避免首个监测样本因惰性加载产生明显延迟（失败仅提示）
    try:
        models._ensure_loaded()
    except Exception as exc:
        print(f"[startup] 模型未就绪: {exc}")
    # 默认以"正常"场景启动监测，打开看板即有实时数据
    engine.start("normal")
    yield
    engine.stop()
    db.close()


app = FastAPI(
    title="轴承故障预测性维护系统",
    description="基于物理仿真信号 + 机器学习的滚动轴承故障诊断与剩余寿命预测",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(monitor.router)
app.include_router(diagnose.router)
app.include_router(history.router)


@app.get("/api/health", tags=["system"])
def health(request: Request):
    """健康检查：服务状态与模型就绪情况。"""
    return {"status": "ok", "models_ready": request.app.state.models.ready}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


# 静态资源（CSS / JS / ECharts 本地文件）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT)
