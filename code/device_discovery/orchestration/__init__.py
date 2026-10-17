"""设备发现编排层的稳定公开入口。"""

from __future__ import annotations

import threading

from device_discovery.collector import DeviceCollector, get_device_collector
from device_discovery.history import DeviceHistoryStore, get_history_store
from device_discovery.rules import RuleStore, get_rule_store

from ._service import DeviceDiscoveryOrchestrator, FailureCallback, ResultCallback
from .errors import (
    OrchestrationError,
    OrchestratorScanError,
    OrchestratorStartError,
    OrchestratorStateError,
    OrchestratorStopError,
)
from .models import (
    FailureStage,
    IgnoreReason,
    IgnoredDevice,
    ProcessingFailure,
    ProcessingOutcome,
    RecognizedDevice,
    RuleChangeReport,
    ScanReport,
)

_default_orchestrator: DeviceDiscoveryOrchestrator | None = None
_default_orchestrator_lock = threading.Lock()


def create_orchestrator(
    *,
    rule_store: RuleStore | None = None,
    collector: DeviceCollector | None = None,
    history_store: DeviceHistoryStore | None = None,
) -> DeviceDiscoveryOrchestrator:
    """使用显式依赖或三个模块的默认共享实例创建编排器。"""
    return DeviceDiscoveryOrchestrator(
        rule_store=get_rule_store() if rule_store is None else rule_store,
        collector=get_device_collector() if collector is None else collector,
        history_store=get_history_store() if history_store is None else history_store,
    )


def get_orchestrator() -> DeviceDiscoveryOrchestrator:
    """返回当前进程共享的默认编排器。"""
    global _default_orchestrator
    if _default_orchestrator is None:
        with _default_orchestrator_lock:
            if _default_orchestrator is None:
                _default_orchestrator = create_orchestrator()
    return _default_orchestrator


__all__ = [
    "DeviceDiscoveryOrchestrator",
    "FailureCallback",
    "FailureStage",
    "IgnoreReason",
    "IgnoredDevice",
    "OrchestrationError",
    "OrchestratorScanError",
    "OrchestratorStartError",
    "OrchestratorStateError",
    "OrchestratorStopError",
    "ProcessingFailure",
    "ProcessingOutcome",
    "RecognizedDevice",
    "ResultCallback",
    "RuleChangeReport",
    "ScanReport",
    "create_orchestrator",
    "get_orchestrator",
]
