"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class MonitorStart(BaseModel):
    scenario: str = Field(
        "normal",
        description="监测场景: normal / inner / outer / ball / degradation")


class DiagnoseRequest(BaseModel):
    samples: List[float] = Field(
        ..., description="振动加速度信号（一维数组，建议 6000 点 = 0.5s @12kHz）")
    sample_rate: float = Field(12000.0, gt=0, description="采样频率 Hz")
    shaft_freq: float | None = Field(
        None, gt=0, description="转频 Hz；留空则从信号频谱自动估计转速")


class DiagnoseResponse(BaseModel):
    fault_type: str
    fault_name_zh: str
    confidence: float
    severity: float
    rul_hours: float
    probs: dict[str, float]
